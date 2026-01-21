import sys
import os
import argparse
import torch
import numpy as np
import pandas as pd
import h5pickle as h5py
from tqdm import tqdm
from peft import PeftModel
from transformers import AutoTokenizer
import random
import json

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 引入必要组件
from fine_tuning import (
    DESCRIPTION_LEN, 
    SIGNAL_TOKEN_ID, 
    get_bearllm, 
    mod_xt_for_qwen, 
    HyperParameters,
    DEFAULT_QWEN_WEIGHTS
)

# 10分类标签映射 (CWRU)
LABEL_MAP = {
    0: "Normal",
    1: "Ball_Mild",
    2: "Ball_Moderate",
    3: "Ball_Severe",
    4: "Inner_Mild",
    5: "Inner_Moderate",
    6: "Inner_Severe",
    7: "Outer_Mild",
    8: "Outer_Moderate",
    9: "Outer_Severe"
}

def str2bool(v):
    if isinstance(v, bool): return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'): return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'): return False
    else: raise argparse.ArgumentTypeError('Boolean value expected.')

def parse_label(output_text: str) -> int:
    """ 解析 LLM 输出文本为 Label ID """
    for lbl_id, lbl_name in LABEL_MAP.items():
        if lbl_name in output_text:
            return lbl_id
    if "Normal" in output_text: return 0
    return -1

def run_classification(model, tokenizer, device, vib_data, instruction):
    # 临时保存信号 (单通道)
    np.save('./cache.npy', vib_data)

    place_holder_ids = torch.ones(DESCRIPTION_LEN, dtype=torch.long) * SIGNAL_TOKEN_ID
    text_part1, text_part2 = mod_xt_for_qwen(instruction)

    user_part1_ids = tokenizer(text_part1, return_tensors='pt', add_special_tokens=False).input_ids[0]
    user_part2_ids = tokenizer(text_part2, return_tensors='pt', add_special_tokens=False).input_ids[0]
    
    user_ids = torch.cat([user_part1_ids, place_holder_ids, user_part2_ids]).to(device)
    attention_mask = torch.ones_like(user_ids).to(device)

    with torch.no_grad():
        output = model.generate(
            user_ids.unsqueeze(0),
            attention_mask=attention_mask.unsqueeze(0),
            max_new_tokens=64
        )

    output_text = tokenizer.decode(output[0, user_ids.shape[0]:], skip_special_tokens=True)
    return output_text

def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading test data from {args.test_json}...")
    
    with open(args.test_json, 'r') as f:
        data_list = json.load(f)

    if args.sample_ratio < 1.0:
        n_samples = int(len(data_list) * args.sample_ratio)
        data_list = random.sample(data_list, n_samples)
        print(f"Sampled {n_samples} items for testing.")

    # 准备模型配置
    hp = HyperParameters()
    hp.device = device
    hp.qwen_weights = args.qwen_weights
    hp.fcn_weights = args.fcn_weights 
    hp.num_classes = args.num_classes
    hp.use_mecs = args.use_mecs 
    # [Fix 1] 关键修复：传递 mecs_type
    hp.mecs_type = args.mecs_type 

    # 加载 Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.qwen_weights, trust_remote_code=True)
    tokenizer.pad_token_id = tokenizer.eos_token_id

    # 加载模型 (Base + LoRA)
    print(f"Loading model... MECS={hp.use_mecs} | Type={hp.mecs_type}")
    base_model = get_bearllm(hp, train_mode=False) 
    
    adapter_path = os.path.join(args.lora_weights, 'vibration_adapter.pth')
    if os.path.exists(adapter_path):
        print(f"Loading Adapter weights from {adapter_path}...")
        adapter_state = torch.load(adapter_path, map_location=device)
        
        # 清洗字典键名
        new_state_dict = {}
        for k, v in adapter_state.items():
            new_key = k.replace(".base_layer", "").replace(".default", "")
            if "lora_" in new_key: continue 
            new_state_dict[new_key] = v
            
        # 尝试加载
        try:
            base_model.get_input_embeddings().adapter.alignment_layer.load_state_dict(new_state_dict, strict=False)
            print("Adapter loaded successfully.")
        except Exception as e:
            print(f"Warning loading adapter: {e}")

    else:
        # 如果找不到 adapter，不要报错退出，因为可能是 Base 模型测试
        print(f"Warning: Adapter weights not found at {adapter_path}. Testing with Base Model.")

    print(f"Loading LoRA from {args.lora_weights}...")
    try:
        model = PeftModel.from_pretrained(base_model, args.lora_weights)
    except Exception as e:
        print(f"Could not load LoRA (maybe only Adapter exists?): {e}")
        model = base_model # Fallback
        
    model.to(device)
    model.eval()

    # 确定测试标签范围
    valid_test_labels = []
    if args.class_order:
         valid_test_labels = [int(x) for x in args.class_order.split(',')]
         print(f"Restricting test to classes: {valid_test_labels}")

    # 读取数据
    try:
        f = h5py.File(args.hdf5_path, 'r')
        vib_dataset = f['vibration']
    except Exception as e:
        print(f"Error opening HDF5: {e}")
        return

    results = []
    all_true_labels, all_pred_labels = [], []

    # 构造 Prompt 的类别列表
    # 只列出当前测试涉及的类别，或者全部类别 (取决于您的设定，通常给全部让它选更难)
    all_classes_str = ",".join(LABEL_MAP.values())
    instruction_template = f"Based on the provided bearing state description #state_place_holder#, identify the type of fault from [{all_classes_str}]."

    print("Starting inference...")
    try:
        for entry in tqdm(data_list):
            vib_id = entry['file_id']
            true_label = entry['label']

            # 过滤不需要的类别
            if valid_test_labels and true_label not in valid_test_labels:
                continue
            
            if str(vib_id) not in vib_dataset:
                continue

            vib_data_raw = vib_dataset[str(vib_id)][:]
            vib_data = vib_data_raw[np.newaxis, :].astype(np.float32)

            output_text = run_classification(model, tokenizer, device, vib_data, instruction_template)
            pred_label = parse_label(output_text)

            results.append({
                "vib_id": vib_id,
                "true_label": LABEL_MAP.get(true_label, "Unknown"),
                "pred_label": LABEL_MAP.get(pred_label, "Unknown"),
                "true_label_id": true_label,
                "pred_label_id": pred_label,
                "raw_output": output_text
            })

            all_true_labels.append(true_label)
            all_pred_labels.append(pred_label)
    finally:
        f.close()

    df = pd.DataFrame(results)
    df.to_csv(args.save_csv, index=False)
    print(f"Results saved to {args.save_csv}")

    if not all_pred_labels:
        print("Warning: No valid predictions extracted.")
        return

    acc = np.mean([t == p for t, p in zip(all_true_labels, all_pred_labels)])
    print(f"\n>>> Test Accuracy: {acc * 100:.2f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--hdf5_path', type=str, default="/home/hzm/CLI-LLM/data/CRWU_data.hdf5")
    parser.add_argument('--test_json', type=str, default='/home/hzm/CLI-LLM/data/CRWU_data_test_ids.json')
    parser.add_argument('--lora_weights', type=str, required=True)
    parser.add_argument('--fcn_weights', type=str, default='/home/hzm/CLI-LLM/LLM/F2LNet_LLM_weight/F2LNet_CRWU')
    parser.add_argument('--qwen_weights', type=str, default=DEFAULT_QWEN_WEIGHTS)
    parser.add_argument('--save_csv', type=str, default="/home/hzm/CLI-LLM/data/CRWU_test_results.csv")
    parser.add_argument('--sample_ratio', type=float, default=1.0)
    parser.add_argument('--num_classes', type=int, default=10)
    parser.add_argument('--use_mecs', type=str2bool, default=True)
    parser.add_argument('--class_order', type=str, default="", help="Limit test to specific classes")
    
    # [Fix 2] 必须添加这个参数，否则消融实验无法生效
    parser.add_argument('--mecs_type', type=str, default="full", 
                        choices=['full', 'b1', 'b2', 'b3', 'b4', 'b5', 'b6'])
    
    args = parser.parse_args()
    evaluate(args)