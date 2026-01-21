import os
import json
import random
import torch
import numpy as np
import sqlite3
import h5pickle as h5py
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
# [修改] 引入 train_id_path 以便读取当前的训练集列表
from datasets import create_cache_dataset, meta_path, data_path, train_id_path
from F2LNet import F2LNet

# === 全局配置 ===
MEMORY_FILE = "memory_bank.json"
BASE_TRAIN_NUM = 200   
BASE_TEST_NUM = 200    
INC_TEST_NUM = 200     

class RawDataset(Dataset):
    def __init__(self, file_ids):
        self.file_ids = file_ids
        self.h5 = h5py.File(data_path, 'r')['vibration']
    def __len__(self): return len(self.file_ids)
    def __getitem__(self, idx):
        fid = self.file_ids[idx]
        data = self.h5[str(fid)][:]
        data = data[np.newaxis, :].astype(np.float32)
        return torch.tensor(data), fid

def get_ids_by_class(label):
    """
    从全量数据库获取某类的所有ID (仅用于 prep 阶段生成初始训练集)
    """
    conn = sqlite3.connect(meta_path)
    c = conn.cursor()
    c.execute(f'SELECT file_id FROM file_info WHERE label = {label}')
    res = [x[0] for x in c.fetchall()]
    conn.close()
    return res

def str2bool(v):
    if isinstance(v, bool): return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'): return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'): return False
    else: raise argparse.ArgumentTypeError('Boolean value expected.')

# ----------------- 功能 1: 准备数据 (Prep) -----------------
def prepare_data(args):
    new_classes = [int(x) for x in args.new_classes.split(',')]
    train_ids = []
    test_ids = []
    
    memory_bank = {}
    if os.path.exists(MEMORY_FILE):
        with open(MEMORY_FILE, 'r') as f:
            memory_bank = {int(k): v for k, v in json.load(f).items()}
    
    is_base = (len(memory_bank) == 0)
    all_seen_classes = list(set(list(memory_bank.keys()) + new_classes))
    current_shots = args.shots
    
    print(f"Dataset Prep | Mode: {'BASE' if is_base else 'INCREMENTAL'} | New: {new_classes}")

    # --- A. 构建训练集 ---
    for c in all_seen_classes:
        if c in new_classes:
            # 新类数据：从全量库中取出指定 shot 数量
            count = BASE_TRAIN_NUM if is_base else current_shots
            all_ids = get_ids_by_class(c)
            random.shuffle(all_ids)
            selected = all_ids[:count] 
            
            # [Trick] 少样本过采样，防止 Linear 层崩溃
            if not is_base: 
                if len(selected) <= 15: selected = selected * 4 
                elif len(selected) <= 35: selected = selected * 2 
            
            train_ids.extend(selected)
        else:
            # 旧类回放 (从记忆库读取)
            replayed = memory_bank[c]
            train_ids.extend(replayed)

    # --- B. 构建测试集 ---
    for c in all_seen_classes:
        all_ids = get_ids_by_class(c)
        unique_train_ids = set(train_ids)
        # 严防泄露：测试集 = 全集 - 训练集(含回放)
        pool = list(set(all_ids) - unique_train_ids)
        
        random.shuffle(pool)
        count = BASE_TEST_NUM if is_base else INC_TEST_NUM
        test_ids.extend(pool[:count])

    create_cache_dataset(
        target_labels=all_seen_classes,
        train_ids_list=train_ids,
        test_ids_list=test_ids
    )

# ----------------- 功能 2: 更新知识库 (Update) -----------------
def update_memory(args):
    """
    更新记忆库
    修改逻辑：严格从当前训练集中挑选样本存入记忆库，防止数据泄露。
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 初始化模型用于提取特征 (仅 Herding 需要，但为了统一逻辑都加载)
    model = F2LNet(num_classes=args.total_classes, use_mecs=args.use_mecs, use_cosine=args.use_cosine).to(device)
    if args.memory_size > 0 and args.strategy == 'herding':
        print(f"Loading weights for Herding from: {args.model_path}")
        model.load_weights(args.model_path)
        model.eval()
    
    memory_bank = {}
    if os.path.exists(MEMORY_FILE):
        with open(MEMORY_FILE, 'r') as f:
            memory_bank = {int(k): v for k, v in json.load(f).items()}

    # [新增] 读取刚刚生成的训练集 ID (CRWU_data_train_ids.json)
    if not os.path.exists(train_id_path):
        raise FileNotFoundError(f"Train ID file not found at {train_id_path}. Run prep first.")
    
    with open(train_id_path, 'r') as f:
        train_records = json.load(f)
    
    # 构建临时查找表：Label -> List of File IDs (仅包含当前训练集样本)
    current_train_pool = {}
    for record in train_records:
        lbl = int(record['label'])
        fid = record['file_id']
        if lbl not in current_train_pool:
            current_train_pool[lbl] = []
        # 注意：prep 阶段可能因为过采样导致 file_id 重复，这里我们去重，因为存记忆库不需要重复样本
        if fid not in current_train_pool[lbl]:
            current_train_pool[lbl].append(fid)

    target_classes = [int(x) for x in args.target_classes.split(',')]
    mem_size = args.memory_size 
    
    print(f"Updating Memory Bank... Strategy: {args.strategy}, Max Size: {mem_size}")
    
    if mem_size == 0:
        print("⚠️ Warning: Memory Size is 0. No samples will be saved.")

    for c in target_classes:
        if c in memory_bank: continue # 已存在的旧类不更新
            
        # [修改] 关键点：只从当前训练池中获取候选样本
        candidates = current_train_pool.get(c, [])
        
        if len(candidates) == 0:
            print(f"⚠️ Warning: No training samples found for Class {c}, cannot update memory.")
            continue

        selected = []

        if mem_size > 0:
            if args.strategy == 'random':
                # === 策略 1: 随机采样 ===
                print(f"  [Random] Selecting from {len(candidates)} candidates for Class {c}...")
                random.shuffle(candidates)
                # 如果 candidates (10) < mem_size (20)，则 k=10，只存 10 个
                selected = candidates[:mem_size]
            
            else: # 'herding'
                # === 策略 2: Herding 精英采样 ===
                print(f"  [Herding] Calculating centers from {len(candidates)} candidates for Class {c}...")
                dataset = RawDataset(candidates)
                loader = DataLoader(dataset, batch_size=64, shuffle=False)
                
                feats_list = []
                ids_list = []
                
                with torch.no_grad():
                    for data, fids in loader:
                        data = data.to(device)
                        feat = model.encoder(data) 
                        feat = F.normalize(feat, dim=1) 
                        feats_list.append(feat.cpu())
                        ids_list.extend(fids.numpy())
                
                if len(feats_list) > 0:
                    feats = torch.cat(feats_list, dim=0)
                    class_mean = torch.mean(feats, dim=0, keepdim=True)
                    class_mean = F.normalize(class_mean, dim=1)
                    dists = torch.norm(feats - class_mean, dim=1)
                    
                    # 选择距离中心最近的 k 个
                    k = min(len(candidates), mem_size)
                    _, top_idx = torch.sort(dists)
                    selected = [int(ids_list[i]) for i in top_idx[:k]]
                else:
                    selected = []

        memory_bank[c] = selected
        print(f"  -> Saved {len(selected)} samples for Class {c}")

    with open(MEMORY_FILE, 'w') as f:
        json.dump(memory_bank, f, indent=4)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='mode')
    
    p_prep = subparsers.add_parser('prep')
    p_prep.add_argument('--new_classes', type=str, required=True)
    p_prep.add_argument('--shots', type=int, default=20)
    
    p_upd = subparsers.add_parser('update')
    p_upd.add_argument('--target_classes', type=str, required=True)
    p_upd.add_argument('--total_classes', type=int, default=10)
    p_upd.add_argument('--model_path', type=str, required=True)
    
    p_upd.add_argument('--strategy', type=str, default='herding', choices=['random', 'herding'])
    p_upd.add_argument('--use_mecs', type=str2bool, default=True)
    p_upd.add_argument('--use_cosine', type=str2bool, default=True)
    p_upd.add_argument('--memory_size', type=int, default=20, help="Number of samples to store per class")
    
    args = parser.parse_args()
    
    if args.mode == 'prep':
        prepare_data(args)
    elif args.mode == 'update':
        update_memory(args)