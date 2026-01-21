import json
import random
import os
import sqlite3
# Import dynamic paths from datasets
from datasets import train_id_path, corpus_path as output_json

num_per_task = 5000 

label_map = {
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

# # 10-Class Label Map xxx
# label_map = {
#     0: "Normal",
#     1: "Single_Pitting_Mild",      # C01: 3点 (轻度)
#     2: "Single_Pitting_Moderate",  # C02: 6点 (中度)
#     3: "Single_Pitting_Severe",    # C03: 9点 (重度)
#     4: "Single_Crack_Mild",        # C04: 0.5mm (轻度)
#     5: "Single_Crack_Severe",      # C05: 2.0mm (重度 - 深度最深)
#     6: "Double_Pitting_Mild",      # C06: 3点 (轻度)
#     7: "Double_Pitting_Moderate",  # C07: 6点 (中度)
#     8: "Double_Crack_Mild",        # C08: 0.5mm (轻度)
#     9: "Double_Crack_Moderate",    # C09: 1.0mm (中度 - 比C05浅，比C08深)
# }

task_name_map = {
    0: "Fault Detection (Binary)",
    1: "Fault Diagnosis (Multi-class)",
    3: "Maintenance Advice",
    4: "Comprehensive Analysis"
}

def generate_maintenance_advice(label_name):
    base_template = (
        f"**Condition Assessment:** The bearing is identified as **{label_name}**.\n\n"
        "**Analysis:** "
    )
    specific_advice = ""
    if "Normal" in label_name:
        specific_advice = (
            "Vibration signatures are within baseline limits. No mechanical defects detected.\n"
            "**Recommended Actions:**\n"
            "* Continue routine monitoring schedule.\n"
            "* Maintain current lubrication intervals."
        )
    elif "Ball" in label_name:
        specific_advice = (
            "Defective rolling elements detected. This causes unstable vibration and noise, potentially leading to cage damage.\n"
            "**Risks:** Variable vibration levels, potential jamming.\n"
            "**Recommended Actions:**\n"
            "* Schedule replacement at next maintenance window.\n"
            "* Check grease for metal particles."
        )
    elif "Inner" in label_name:
        specific_advice = (
            "Inner race defect detected. Characterized by high-frequency impacts modulated by shaft speed.\n"
            "**Risks:** Shaft damage, rapid degradation.\n"
            "**Recommended Actions:**\n"
            "* Reduce load if possible.\n"
            "* Plan for immediate replacement."
        )
    elif "Outer" in label_name:
        specific_advice = (
            "Outer race defect detected. Shows consistent impact patterns, stationary relative to load zone.\n"
            "**Risks:** Housing damage, increased noise.\n"
            "**Recommended Actions:**\n"
            "* Inspect housing fit and alignment.\n"
            "* Replace bearing."
        )
    else:
        specific_advice = "Unknown condition. Please check sensor data."

    return base_template + specific_advice

def generate_tasks(data, num_per_task=5000):
    corpus = []
    all_ids = list(range(len(data)))
    random.shuffle(all_ids)

    target_tasks = [1]
    task_samples = {tid: random.sample(all_ids, min(num_per_task, len(all_ids))) for tid in target_tasks}
    
    task_counter = 0
    all_classes_str = ",".join(label_map.values())

    if 0 in target_tasks:
        print(f"Generating Task 0 - {task_name_map[0]}")
        for idx in task_samples[0]:
            entry = data[idx]
            label = entry['label']
            corpus.append({
                "id": task_counter,
                "task_id": 0,
                "instruction": "Based on the provided bearing state description #state_place_holder#, assess whether the bearing is experiencing any faults. Provide 'yes' or 'no'.",
                "label_id": label,
                "vib_id": entry['file_id'],
                "response": "no" if label == 0 else "yes",
                "condition_id": entry['condition_id']
            })
            task_counter += 1

    if 1 in target_tasks:
        print(f"Generating Task 1 - {task_name_map[1]}")
        for idx in task_samples[1]:
            entry = data[idx]
            label_str = label_map[entry['label']]
            corpus.append({
                "id": task_counter,
                "task_id": 1,
                "instruction": f"Based on the provided bearing state description #state_place_holder#, identify the type of fault from [{all_classes_str}].",
                "label_id": entry['label'],
                "vib_id": entry['file_id'],
                "response": label_str,
                "condition_id": entry['condition_id']
            })
            task_counter += 1

    if 3 in target_tasks:
        print(f"Generating Task 3 - {task_name_map[3]}")
        for idx in task_samples[3]:
            entry = data[idx]
            label_str = label_map[entry['label']]
            corpus.append({
                "id": task_counter,
                "task_id": 3,
                "instruction": "Based on the provided bearing state description #state_place_holder#, provide detailed maintenance advice for the bearing.",
                "label_id": entry['label'],
                "vib_id": entry['file_id'],
                "response": generate_maintenance_advice(label_str),
                "condition_id": entry['condition_id']
            })
            task_counter += 1

    if 4 in target_tasks:
        print(f"Generating Task 4 - {task_name_map[4]}")
        for idx in task_samples[4]:
            entry = data[idx]
            label_str = label_map[entry['label']]
            corpus.append({
                "id": task_counter,
                "task_id": 4,
                "instruction": f"Based on the provided bearing state description #state_place_holder#, identify the type of fault from [{all_classes_str}], and provide detailed maintenance advice.",
                "label_id": entry['label'],
                "vib_id": entry['file_id'],
                "response": generate_maintenance_advice(label_str), 
                "condition_id": entry['condition_id']
            })
            task_counter += 1

    return corpus

if __name__ == "__main__":
    if not os.path.exists(train_id_path):
        print(f"Error: {train_id_path} not found! Please run datasets.py first.")
        exit()

    print(f"Loading training set data from {train_id_path}...")
    with open(train_id_path, 'r') as f:
        data = json.load(f)
    print(f"Total samples in training set: {len(data)}")

    print("Generating tasks...")
    corpus = generate_tasks(data, num_per_task=num_per_task)
    print(f"Total generated corpus size: {len(corpus)}")

    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    with open(output_json, "w") as f:
        json.dump(corpus, f, indent=4)
    print(f"Corpus saved to {output_json}")


