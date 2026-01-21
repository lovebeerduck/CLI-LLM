import os
import json
import h5py
import sqlite3
import numpy as np
import scipy.io as sio

# ---------------- 配置 -----------------
raw_data_root = "/home/hzm/CIL-LLM/DATA/CRWU"
output_dir = "/home/hzm/CIL-LLM/DATA"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

output_h5 = os.path.join(output_dir, "CRWU_data.hdf5")
meta_file = os.path.join(output_dir, "CRWU_dataset_meta.json")
sqlite_file = os.path.join(output_dir, "CRWU_data.sqlite")

TARGET_LEN = 4096 

# ---------------- 预处理 -----------------
def pad_or_cut(data: np.ndarray, length=4096):
    if len(data) < length:
        data = np.pad(data, (0, length - len(data)))
    else:
        data = data[:length]
    return data

def normalize(data: np.ndarray):
    mean = np.mean(data)
    std = np.std(data)
    if std == 0:
        return (data - mean).astype(np.float32)
    return ((data - mean) / std).astype(np.float32)

def preprocess_signal(data: np.ndarray, length=4096):
    data = pad_or_cut(data, length)
    data = normalize(data)
    return data

# ---------------- 映射字典 -----------------
fault_label_dict = {
    "Normal Baseline": 0,
    "Ball_0007": 1, "Ball_0014": 2, "Ball_0021": 3,
    "Inner Race_0007": 4, "Inner Race_0014": 5, "Inner Race_0021": 6,
    "Outer Race_0007": 7, "Outer Race_0014": 8, "Outer Race_0021": 9,
}

condition_id_dict = {"DE": 1, "FE": 2, "BA": 3}

# ---------------- 处理逻辑 -----------------
def process_crwu(root_folder, output_h5, meta_file, sqlite_file, target_len=4096):
    meta = {}
    data_segments = []
    label_map = {}   # file_id -> (label, condition_id)  <-- 移除 rul
    file_id = 0

    for fault_folder in ["Normal Baseline", "Ball", "Inner Race", "Outer Race"]:
        fault_path = os.path.join(root_folder, fault_folder)
        if not os.path.exists(fault_path):
            continue

        if fault_folder == "Normal Baseline":
            load_folders = [""]
        else:
            load_folders = [d for d in os.listdir(fault_path) if os.path.isdir(os.path.join(fault_path, d))]

        for load in load_folders:
            path = os.path.join(fault_path, load) if load else fault_path
            if not os.path.isdir(path): continue

            mat_files = [f for f in os.listdir(path) if f.endswith(".mat")]
            if not mat_files: continue

            v_ids = []
            for mat_file in mat_files:
                try:
                    mat_data = sio.loadmat(os.path.join(path, mat_file))
                except Exception:
                    continue

                # 查找通道 (更稳健的匹配逻辑)
                channels = {}
                for k in mat_data.keys():
                    if k.startswith("__") or "RPM" in k: continue
                    k_lower = k.lower()
                    if "de" in k_lower: channels["DE"] = mat_data[k].squeeze()
                    elif "fe" in k_lower: channels["FE"] = mat_data[k].squeeze()
                    elif "ba" in k_lower: channels["BA"] = mat_data[k].squeeze()

                for ch, signal in channels.items():
                    slices = [signal[i:i+target_len] for i in range(0, len(signal), target_len)]
                    for seg in slices:
                        if len(seg) < target_len: continue
                        
                        seg_processed = preprocess_signal(seg, target_len)
                        data_segments.append(seg_processed)

                        if fault_folder == "Normal Baseline":
                            label = 0
                        else:
                            key = f"{fault_folder}_{load}"
                            label = fault_label_dict.get(key, -1)
                            if label == -1: continue

                        condition_id = condition_id_dict.get(ch, 1)
                        
                        # --- 修改：只存 label 和 condition_id，不存 rul ---
                        label_map[file_id] = (label, condition_id)
                        v_ids.append(file_id)
                        file_id += 1

            meta_key = f"{fault_folder}_{load}" if load else fault_folder
            if v_ids:
                meta[meta_key] = {"ids": [v_ids[0], v_ids[-1]]}

    print(f"Total samples: {len(data_segments)}")

    # 写入 HDF5
    with h5py.File(output_h5, "w") as h5f:
        vib_group = h5f.create_group("vibration")
        for idx, segment in enumerate(data_segments):
            vib_group.create_dataset(str(idx), data=segment)

    # 写入 Meta
    with open(meta_file, "w") as f:
        json.dump(meta, f, indent=4)

    # 写入 SQLite
    create_sqlite(label_map, sqlite_file)
    return label_map

def create_sqlite(label_map, sqlite_file):
    if os.path.exists(sqlite_file):
        os.remove(sqlite_file)
        
    conn = sqlite3.connect(sqlite_file)
    cur = conn.cursor()

    cur.execute("""
    CREATE TABLE IF NOT EXISTS condition (
        condition_id INTEGER PRIMARY KEY,
        dataset TEXT,
        code TEXT,
        channel TEXT
    )
    """)
    
    # --- 修改：移除了 rul 列 ---
    cur.execute("""
    CREATE TABLE IF NOT EXISTS file_info (
        file_id INTEGER PRIMARY KEY,
        condition_id INTEGER,
        label INTEGER
    )
    """)
    
    cur.execute("""
    CREATE TABLE IF NOT EXISTS label_note (
        label INTEGER PRIMARY KEY,
        note TEXT
    )
    """)

    cur.execute("INSERT OR IGNORE INTO condition VALUES (1,'CRWU','CWRU_DE','Drive End')")
    cur.execute("INSERT OR IGNORE INTO condition VALUES (2,'CRWU','CWRU_FE','Fan End')")
    cur.execute("INSERT OR IGNORE INTO condition VALUES (3,'CRWU','CWRU_BA','Base')")

    label_note_dict = {
        0: "Normal",
        1: "Ball_Mild", 2: "Ball_Moderate", 3: "Ball_Severe",
        4: "Inner_Mild", 5: "Inner_Moderate", 6: "Inner_Severe",
        7: "Outer_Mild", 8: "Outer_Moderate", 9: "Outer_Severe",
    }
    for k, v in label_note_dict.items():
        cur.execute("INSERT OR IGNORE INTO label_note(label,note) VALUES (?,?)", (k, v))

    # --- 修改：写入时不包含 rul ---
    data_list = [(fid, condition_id, label) for fid, (label, condition_id) in label_map.items()]
    cur.executemany("INSERT INTO file_info(file_id,condition_id,label) VALUES (?,?,?)", data_list)

    conn.commit()
    conn.close()

if __name__ == "__main__":
    process_crwu(raw_data_root, output_h5, meta_file, sqlite_file, target_len=TARGET_LEN)
    print("✅ CRWU 数据处理完成 (无RUL版)")