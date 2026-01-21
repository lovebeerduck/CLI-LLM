import sqlite3
import random
import json
import h5pickle as h5py
import os
from torch.utils.data import Dataset, DataLoader
import numpy as np

# ---------------- Path Configuration -----------------
# Check for environment variable to support switching datasets dynamically
BASE_DATA_DIR = os.environ.get('CWRU_DATA_DIR', '/home/hzm/CLI-LLM/data')

# Construct paths relative to the base directory
# Assumes the files inside the length folders are named 'CRWU_data.hdf5', etc.
data_path = os.path.join(BASE_DATA_DIR, 'CRWU_data.hdf5')
meta_path = os.path.join(BASE_DATA_DIR, 'CRWU_data.sqlite')
cache_path = os.path.join(BASE_DATA_DIR, 'CRWU_data.json')
corpus_path = os.path.join(BASE_DATA_DIR, 'CRWU_data_corpus.json')
train_id_path = os.path.join(BASE_DATA_DIR, 'CRWU_data_train_ids.json')
test_id_path = os.path.join(BASE_DATA_DIR, 'CRWU_data_test_ids.json')

# data_path = '/home/hzm/CIL-LLM/data/XXX_data.hdf5'
# meta_path = '/home/hzm/CIL-LLM/data/XXX_data.sqlite'
# cache_path = '/home/hzm/CIL-LLM/data/XXX_data.json'
# corpus_path = '/home/hzm/CIL-LLM/data/XXX_data_corpus.json'
# train_id_path = '/home/hzm/CIL-LLM/data/XXX_data_train_ids.json'
# test_id_path = '/home/hzm/CIL-LLM/data/XXX_data_test_ids.json'

def get_ref_ids():
    conn = sqlite3.connect(meta_path)
    cursor = conn.cursor()
    cursor.execute('SELECT condition_id, file_id FROM file_info WHERE label = 0')
    ref_data = cursor.fetchall()
    conn.close()
    ref_ids = {}
    for condition_id, file_id in ref_data:
        ref_ids.setdefault(condition_id, []).append(file_id)
    return ref_ids

def create_cache_dataset(target_labels=None, sample_ratio_dict=None, train_ids_list=None, test_ids_list=None):
    print(f"Creating dataset cache in {BASE_DATA_DIR}... Labels: {target_labels}")
    
    conn = sqlite3.connect(meta_path)
    cursor = conn.cursor()
    cursor.execute('SELECT condition_id, file_id, label FROM file_info')
    all_data = cursor.fetchall()
    conn.close()

    strict_mode = (train_ids_list is not None)
    valid_train = set(train_ids_list) if train_ids_list else set()
    valid_test = set(test_ids_list) if test_ids_list else set()
    
    if strict_mode:
        print(f"Mode: STRICT (Train: {len(valid_train)}, Test: {len(valid_test)})")
    else:
        print(f"Mode: RATIO Sampling ({sample_ratio_dict})")

    label_groups = {}
    dataset_info = {subset: [] for subset in ['train', 'val', 'test']}
    train_id_records = []   
    test_id_records = []  

    for condition_id, file_id, label in all_data:
        if target_labels is not None and label not in target_labels:
            continue
        
        if strict_mode:
            record = {"file_id": file_id, "label": label, "condition_id": condition_id}
            if file_id in valid_train:
                dataset_info['train'].append([file_id, label])
                train_id_records.append(record)
            elif file_id in valid_test:
                dataset_info['test'].append([file_id, label])
                dataset_info['val'].append([file_id, label]) 
                test_id_records.append(record)
        else:
            if label not in label_groups:
                label_groups[label] = []
            label_groups[label].append((condition_id, file_id, label))

    if not strict_mode:
        for label, samples in label_groups.items():
            random.shuffle(samples)
            ratio = 1.0
            if sample_ratio_dict and label in sample_ratio_dict:
                ratio = sample_ratio_dict[label]
            keep_n = int(len(samples) * ratio)
            if keep_n < 1: keep_n = 1
            samples = samples[:keep_n]
            
            n_total = len(samples)
            n_train = int(n_total * 0.7)
            n_val = int(n_total * 0.2)
            
            subsets = (
                ('train', samples[:n_train]),
                ('val', samples[n_train:n_train+n_val]),
                ('test', samples[n_train+n_val:])
            )
            
            for subset_name, subset_samples in subsets:
                for condition_id, file_id, label in subset_samples:
                    dataset_info[subset_name].append([file_id, label])
                    record = {"file_id": file_id, "label": label, "condition_id": condition_id}
                    if subset_name == 'train':
                        train_id_records.append(record)
                    else: 
                        test_id_records.append(record)

    with open(cache_path, 'w') as f:
        json.dump(dataset_info, f, indent=2)
    with open(train_id_path, 'w') as f:
        json.dump(train_id_records, f, indent=2)   
    with open(test_id_path, 'w') as f:
        json.dump(test_id_records, f, indent=2)

def load_cache_dataset():
    if not os.path.exists(cache_path):
        print("Warning: Cache not found, creating empty default...")
        create_cache_dataset() 
    with open(cache_path, 'r') as f:
        dataset_info = json.load(f)
    return dataset_info

class VibDataset(Dataset):              
    def __init__(self, subset_info, is_train=False):
        self.subset_info = subset_info
        self.data = h5py.File(data_path, 'r')['vibration']
        self.is_train = is_train

    def __len__(self):
        return len(self.subset_info)

    def _augment(self, x):
        if random.random() < 0.5:
            noise = np.random.normal(0, 0.01, x.shape)
            x = x + noise
        if random.random() < 0.5:
            scale = np.random.uniform(0.9, 1.1)
            x = x * scale
        if random.random() < 0.5:
            shift = np.random.randint(-50, 50)
            x = np.roll(x, shift, axis=1)
        return x

    def __getitem__(self, idx):
        file_id, label = self.subset_info[idx]
        vib_data = self.data[str(file_id)][:]
        data = vib_data[np.newaxis, :].astype(np.float32)
        if self.is_train:
            data = self._augment(data)
        data = data.astype(np.float32) 
        label = int(label)
        return data, label, file_id

def get_loaders(batch_size, num_workers):
    dataset_info = load_cache_dataset()
    train_set = VibDataset(dataset_info['train'], is_train=True) 
    val_set = VibDataset(dataset_info['val'], is_train=False)
    test_set = VibDataset(dataset_info['test'], is_train=False)
    train_loader = DataLoader(train_set, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    return train_loader, val_loader, test_loader

class CorpusDataset:
    def __init__(self):
        self.vib_data = h5py.File(data_path, 'r')['vibration']
        if not os.path.exists(corpus_path):
             raise FileNotFoundError(f"{corpus_path} not found. Run corpus_creat.py first.")
        self.corpus = json.load(open(corpus_path, 'r'))

    def __len__(self):
        return len(self.corpus)

    def __getitem__(self, idx):
        c = self.corpus[idx]
        vib_data = self.vib_data[str(c['vib_id'])][:]
        vib = vib_data[np.newaxis, :].astype(np.float32)
        return c['id'], c.get('task_id', 1), c['label_id'], vib, c['instruction'], c['response'], None


