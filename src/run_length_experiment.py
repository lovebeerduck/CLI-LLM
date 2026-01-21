import os
import subprocess
import sys
import pandas as pd
import time

PYTHON_EXEC = sys.executable 
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))


DATA_LENGTHS = ["4096"]
BASE_DATA_PATH = "/home/hzm/CLI-LLM/data/CWRU"

EXP_PARENT_DIR = os.path.join(ROOT_DIR, "Experiment_Length_Robustness_CWRU_50_30_10")

MECS_TYPE = "full" 
USE_MECS = "True"

TASKS = [
    {"stage": 1, "name": "Task1", "new": "0,1,4,7", "total": 4,  "shots": 0, "epochs": 50}, 
    {"stage": 2, "name": "Task2", "new": "2,5",     "total": 6,  "shots": 50, "epochs": 300},
    {"stage": 3, "name": "Task3", "new": "8,3",     "total": 8,  "shots": 30, "epochs": 300},
    {"stage": 4, "name": "Task4", "new": "6,9",     "total": 10, "shots": 10, "epochs": 300},
]

def run_cmd(cmd, step_name, env=None):
    print(f"\n[{step_name}] Executing: {cmd}")
    # Pass environment variables to the subprocess
    if env is None:
        env = os.environ.copy()
    
    ret = subprocess.call(cmd, shell=True, env=env)
    if ret != 0:
        print(f"❌ Error in {step_name}. Stopping.")
        sys.exit(1)

def get_metrics(save_dir, filename="cnn_metrics.json"):
    import json
    m = {"acc": 0.0}
    try:
        with open(os.path.join(save_dir, filename), 'r') as f: m.update(json.load(f))
    except: pass
    return m

def get_llm_acc(csv_path):
    try:
        df = pd.read_csv(csv_path)
        return (df['true_label'] == df['pred_label']).mean()
    except: return 0.0

def run_length_pipeline(length):
    data_dir = os.path.join(BASE_DATA_PATH, length)
    save_root = os.path.join(EXP_PARENT_DIR, length) 
    
    print(f"\n{'#'*60}")
    print(f"🚀 Starting Experiment for Length: {length}")
    print(f"📂 Data Dir: {data_dir}")
    print(f"💾 Save Dir: {save_root}")
    print(f"{'#'*60}")

    if not os.path.exists(save_root): os.makedirs(save_root)
    

    env = os.environ.copy()
    env["CWRU_DATA_DIR"] = data_dir
    if os.path.exists("memory_bank.json"): os.remove("memory_bank.json")

    results = []
    prev_fcn_dir = None
    prev_adapter_path = None
    cumulative_classes = []

    for task in TASKS:
        stage = task['stage']
        task_name = task['name']
        
        task_dir = os.path.join(save_root, task_name)
        fcn_dir = os.path.join(task_dir, "FCN")
        lora_dir = os.path.join(task_dir, "LoRA")
        llm_csv = os.path.join(task_dir, "llm_result.csv")
        
        # Class Order
        new_cls = [str(x) for x in task['new'].split(',')]
        cumulative_classes.extend(new_cls)
        class_order_str = ",".join(cumulative_classes)
        
        print(f"\n--- Stage {stage}: {task_name} (Classes: {task['total']}) ---")

        # 1. Prep
        cmd = f"{PYTHON_EXEC} utils_memory.py prep --new_classes \"{task['new']}\""
        if stage > 1: cmd += f" --shots {task['shots']}"
        run_cmd(cmd, f"S{stage}-Prep", env)

        # 2. Train CNN
        # Stage 1 学习率稍大，后续微调更小
        lr = 0.0005 if stage == 1 else 0.00005
        cmd = f"{PYTHON_EXEC} pre_training.py --num_classes {task['total']} --save_dir {fcn_dir} --epoch_max {task['epochs']} --lr {lr}"
        #cmd += f" --use_mecs {USE_MECS} --mecs_type {MECS_TYPE} --class_order \"{class_order_str}\""
        cmd += f" --use_mecs {USE_MECS} --mecs_type {MECS_TYPE} --use_cosine False --class_order \"{class_order_str}\""
        if stage > 1: cmd += f" --resume_weights {prev_fcn_dir}"
        run_cmd(cmd, f"S{stage}-CNN", env)

        # 3. Update Memory
        cmd = f"{PYTHON_EXEC} utils_memory.py update --target_classes \"{task['new']}\" --total_classes {task['total']} --model_path {fcn_dir}"
        #cmd += f" --strategy herding --use_mecs {USE_MECS} --memory_size 20"
        cmd += f" --strategy random --use_mecs {USE_MECS} --use_cosine False --memory_size 20"
        run_cmd(cmd, f"S{stage}-Memory", env)

        # 4. Corpus
        run_cmd(f"{PYTHON_EXEC} corpus_creat.py", f"S{stage}-Corpus", env)

        # 5. Fine-tune LLM
        cmd = f"{PYTHON_EXEC} fine_tuning.py --num_classes {task['total']} --fcn_weights {fcn_dir} --output_dir {lora_dir}"
        cmd += f" --num_train_epochs 10 --use_mecs {USE_MECS} --mecs_type {MECS_TYPE}"
        if prev_adapter_path:
             cmd += f" --prev_adapter {prev_adapter_path}"
        run_cmd(cmd, f"S{stage}-FineTune", env)

        # 6. Test LLM
        # IMPORTANT: Pass the specific hdf5/json paths to test.py
        current_adapter = os.path.join(lora_dir, "vibration_adapter.pth") 
        
        test_hdf5 = os.path.join(data_dir, "CRWU_data.hdf5")
        test_json = os.path.join(data_dir, "CRWU_data_test_ids.json")
        
        cmd = f"{PYTHON_EXEC} test.py --num_classes {task['total']} --fcn_weights {fcn_dir} --lora_weights {lora_dir}"
        cmd += f" --use_mecs {USE_MECS} --mecs_type {MECS_TYPE} --class_order \"{class_order_str}\" --save_csv {llm_csv}"
        cmd += f" --hdf5_path {test_hdf5} --test_json {test_json}"
        
        run_cmd(cmd, f"S{stage}-Test", env)

        # Metrics
        cnn_metrics = get_metrics(fcn_dir)
        llm_acc = get_llm_acc(llm_csv)
        results.append({
            "stage": stage,
            "cnn_acc": cnn_metrics['acc'],
            "llm_acc": llm_acc
        })
        
        prev_fcn_dir = fcn_dir
        prev_adapter_path = current_adapter

    # Save summary for this length
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(save_root, "summary.csv"), index=False)
    print(f"\n✅ Finished Length {length}. Summary saved.")
    return results

def main():
    if not os.path.exists(EXP_PARENT_DIR): os.makedirs(EXP_PARENT_DIR)
    print(f"📂 Global Result Directory: {EXP_PARENT_DIR}")

    all_summaries = {}
    
    for length in DATA_LENGTHS:
        try:
            res = run_length_pipeline(length)
            all_summaries[length] = res
        except Exception as e:
            print(f"❌ Critical Failure for Length {length}: {e}")
            
    print("\n\n" + "="*60)
    print("🏁 FINAL EXPERIMENT SUMMARY (ALL LENGTHS)")
    print("="*60)
    
    for length, res in all_summaries.items():
        print(f"\n>> Length: {length}")
        print(f"{'Stage':<6} | {'CNN Acc':<10} | {'LLM Acc':<10}")
        for r in res:
             print(f"{r['stage']:<6} | {r['cnn_acc']:.2%}    | {r['llm_acc']:.2%}")

if __name__ == "__main__":
    main()