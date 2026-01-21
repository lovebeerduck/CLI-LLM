import sys
import os
import argparse
import torch
import numpy as np
from dotenv import dotenv_values
# [Fix] Use factory function for newer peft versions
from peft import LoraConfig, TaskType, get_peft_model, PeftModel
from torch import nn
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, TrainingArguments, Trainer
import torch.nn.functional as F

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import components
from F2LNet import FeatureEncoder, CosineProtoClassifier 
from datasets import CorpusDataset

# Default Paths (Adjust if needed)
DEFAULT_QWEN_WEIGHTS = '/home/hzm/CLI-LLM/LLM/qwen_weight'
DEFAULT_FCN_WEIGHTS = '/home/hzm/CLI-LLM/LLM/F2LNet_LLM_weight/F2LNet_CRWU' 
DEFAULT_LORA_SAVE_DIR = '/home/hzm/CLI-LLM/LLM/F2LNet_LLM_weight/lora_CRWU'

# Constants
DESCRIPTION_LEN = 5
LLM_HIDDEN_SIZE = 1536
SIGNAL_TOKEN_ID = 151925
M_BASE = [4**4, 4**3, 4**2, 4**1, 1]

def str2bool(v):
    if isinstance(v, bool): return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'): return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'): return False
    else: raise argparse.ArgumentTypeError('Boolean value expected.')

class HyperParameters:
    def __init__(self, args=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.r = 4
        self.lora_alpha = 32
        self.lora_dropout = 0.1
        self.per_device_train_batch_size = 20
        self.gradient_accumulation_steps = 4
        self.logging_steps = 20
        self.save_steps = 500
        self.learning_rate = 1e-4
        self.lr_scheduler_type = 'cosine'
        
        # Default Params
        self.num_train_epochs = 10
        self.num_classes = 10
        self.fcn_weights = DEFAULT_FCN_WEIGHTS
        self.qwen_weights = DEFAULT_QWEN_WEIGHTS
        self.prev_adapter = None
        self.use_mecs = True 
        # [Fix 1] Add mecs_type default
        self.mecs_type = 'full'

        if args:
            if hasattr(args, 'num_train_epochs'): self.num_train_epochs = args.num_train_epochs
            self.num_classes = args.num_classes
            self.fcn_weights = args.fcn_weights
            self.qwen_weights = args.qwen_weights
            self.prev_adapter = args.prev_adapter
            self.use_mecs = args.use_mecs
            # [Fix 2] Load mecs_type from args
            if hasattr(args, 'mecs_type'): self.mecs_type = args.mecs_type

# System Prompt
sys_prompt = (
    "As an expert in bearing fault diagnosis with extensive knowledge in mechanical engineering and failure "
    "analysis, you can assess the state of bearings. Based on the description of the bearing state, answer my questions."
)

class AlignmentLayer(nn.Module):
    def __init__(self, hp):
        super().__init__()
        self.hp = hp
        self.linear1 = nn.Linear(1024, 128) 
        self.relu = nn.ReLU()
        self.linear3 = nn.Linear(128, DESCRIPTION_LEN * LLM_HIDDEN_SIZE)  

    def forward(self, x):
        x = x.view(-1, 1024) 
        x = self.linear1(x)
        x = self.relu(x)
        x_out = self.linear3(x)
        x_out = x_out.view(x_out.size(0), DESCRIPTION_LEN, LLM_HIDDEN_SIZE)
        x_out = x_out.to(torch.bfloat16)
        return x_out

    def load_smart(self):
        target_path = self.hp.prev_adapter
        if not target_path or not os.path.exists(target_path):
            return
        print(f"Loading previous Adapter weights from: {target_path}")
        try:
            state = torch.load(target_path, map_location='cpu')
            if 'linear1.weight' in state: 
                self.linear1.weight.data = state['linear1.weight']
                self.linear1.bias.data = state['linear1.bias']
            if 'linear3.weight' in state:
                if state['linear3.weight'].shape == self.linear3.weight.shape:
                    self.linear3.weight.data = state['linear3.weight']
                    self.linear3.bias.data = state['linear3.bias']
                    print("Loaded linear3 weights (Direct-Pass mode).")
                else:
                    print("Linear3 shape mismatch (Structure Upgrade). Initializing linear3 from scratch.")
        except Exception as e:
            print(f"Warning: Failed to load prev adapter: {e}")

@torch.no_grad()
def decode_sample_id(signal_ids_tensor):
    signal_ids_tensor = signal_ids_tensor.view(-1, 5)   
    signal_ids_tensor = signal_ids_tensor - SIGNAL_TOKEN_ID
    m_t = torch.tensor(M_BASE, device=signal_ids_tensor.device).unsqueeze(0) 
    sample_ids = (signal_ids_tensor * m_t).sum(dim=1)  
    return sample_ids

class IdConverter:
    def __init__(self, hp, train_mode=True):
        self.hp = hp
        self.dataset = CorpusDataset() if train_mode else None
        
    @torch.no_grad()
    def get_signal(self, signal_ids_tensor, train_mode=True):
        s = signal_ids_tensor
        if s.dim() == 1: s = s.unsqueeze(0)
        sample_ids = decode_sample_id(s)
        res = []
        if train_mode:
            for sid in sample_ids:
                sid_int = int(sid.item())
                # Ensure index is within bounds
                if sid_int < len(self.dataset):
                    item = self.dataset.__getitem__(sid_int)
                    vib = item[3] 
                    res.append(np.array(vib))
                else:
                    # Fallback for safety
                    res.append(np.zeros((1, 2048)))
        else:
            if os.path.exists('./cache.npy'):
                arr = np.load('./cache.npy')
                res.append(arr)
            else:
                res.append(np.zeros((1, 2048)))
                
        data = np.stack(res, axis=0)
        if data.ndim == 2: data = data[:, np.newaxis, :]
        return torch.tensor(data, dtype=torch.float32, device=self.hp.device)

class AlignmentAdapter(nn.Module):
    def __init__(self, hp):
        super().__init__()
        self.hp = hp
        
        # [Fix 3] Pass mecs_type to FeatureEncoder
        print(f"Initializing Visual Encoder: MECS={hp.use_mecs}, Type={hp.mecs_type}")
        self.feature_encoder = FeatureEncoder(use_mecs=hp.use_mecs, mecs_type=hp.mecs_type)
        
        self.classifier = CosineProtoClassifier(num_classes=hp.num_classes)
        self.alignment_layer = AlignmentLayer(hp)

        print(f"Loading F2LNet weights from {hp.fcn_weights}...")
        if os.path.exists(hp.fcn_weights):
            self.feature_encoder.load_weights(hp.fcn_weights)
            try:
                self.classifier.load_weights(hp.fcn_weights)
            except:
                print("Warning: Classifier weights mismatch, skipping (Not critical for LLM tuning)")
        else:
             print(f"⚠️ Warning: FCN weights file not found: {hp.fcn_weights}")

        self.alignment_layer.load_smart()

        # Freeze Vision Part
        for param in self.feature_encoder.parameters(): param.requires_grad = False
        for param in self.classifier.parameters(): param.requires_grad = False
        for param in self.alignment_layer.parameters(): param.requires_grad = True
        print("★ FeatureEncoder & Classifier FROZEN. Training AlignmentLayer only.")

    def train(self, mode=True):
        super().train(mode) 
        if mode:
            self.feature_encoder.eval() 
            self.classifier.eval()
        return self

    def forward(self, x):
        feat = self.feature_encoder(x)
        if feat.dim() > 2: feat = feat.view(feat.size(0), -1)
        return self.alignment_layer(feat)   

class ModifiedEmbedding(nn.Module):
    def __init__(self, embedding, hp, train_mode=True):
        super().__init__()
        self.embedding = embedding
        self.adapter = AlignmentAdapter(hp)
        self.adapter.to(embedding.weight.device)
        self.signal_converter = IdConverter(hp, train_mode=train_mode)

    def forward(self, x):
        B, T = x.size()
        if (x >= SIGNAL_TOKEN_ID).sum() == 0:
            return self.embedding(x)
            
        safe_ids = x.clone()
        safe_ids[safe_ids >= SIGNAL_TOKEN_ID] = self.embedding.num_embeddings - 1  
        base = self.embedding(safe_ids).to(torch.bfloat16)
        
        for b in range(B):
            mask = (x[b] >= SIGNAL_TOKEN_ID)  
            num_sig = int(mask.sum().item())
            if num_sig == 0: continue
            
            signal_ids_flat = x[b, mask]  
            sig_ids_batch = signal_ids_flat.unsqueeze(0)
            signal_feats = self.signal_converter.get_signal(sig_ids_batch, self.training)  
            aligned = self.adapter(signal_feats).to(torch.bfloat16)
            
            # Map aligned features back to sequence
            # Note: This assumes 1 signal -> DESCRIPTION_LEN tokens mapping logic is handled correctly in data prep
            # Here we just broadcast/assign. The dimension must match.
            # alignment_layer output: [1, DESCRIPTION_LEN, HIDDEN]
            # base[b, mask, :] shape: [DESCRIPTION_LEN, HIDDEN]
            
            if aligned.shape[1] == num_sig:
                 base[b, mask, :] = aligned[0]
            else:
                 # Fallback if dimensions don't align perfectly (e.g. multiple signals in one seq)
                 # This logic depends on how you tokenize the signal placeholders
                 pass 
                 base[b, mask, :] = aligned[0] # Simplest assumption
                 
        return base  
    
    @property
    def weight(self): return self.embedding.weight

def get_bearllm(hp, train_mode=True):
    config = AutoConfig.from_pretrained(f'{hp.qwen_weights}/config.json')
    model = AutoModelForCausalLM.from_pretrained(
        hp.qwen_weights,
        device_map=hp.device,
        torch_dtype="auto",
        trust_remote_code=True,
        config=config
    )
    embedding = model.get_input_embeddings()
    mod_embedding = ModifiedEmbedding(embedding, hp, train_mode=train_mode)
    model.set_input_embeddings(mod_embedding)
    return model

def mod_xt_for_qwen(xt):
    text_part1 = '<|im_start|>system\n' + sys_prompt + '\n<|im_end|><|im_start|>user\n' + xt.split('#state_place_holder#')[0]
    text_part2 = xt.split('#state_place_holder#')[1] + '<|im_end|>\n<|im_start|>assistant\n'
    return text_part1, text_part2

def collate_fn(batch):
    input_ids = torch.nn.utils.rnn.pad_sequence([x['input_ids'] for x in batch], batch_first=True, padding_value=0)
    attention_mask = torch.nn.utils.rnn.pad_sequence([x['attention_mask'] for x in batch], batch_first=True, padding_value=0)
    labels = torch.nn.utils.rnn.pad_sequence([x['labels'] for x in batch], batch_first=True, padding_value=-100)
    task_ids = torch.tensor([x.get('task_id', 1) for x in batch], dtype=torch.long)
    return {'input_ids': input_ids.long(), 'attention_mask': attention_mask.long(), 'labels': labels.long(), 'task_id': task_ids}

def encode_sample_id(x):
    result = []
    remainder = x
    for i in range(len(M_BASE)):
        digit = remainder // M_BASE[i]
        result.append(digit)
        remainder %= M_BASE[i]
    return torch.tensor(result, dtype=torch.int)

class FineTuningDataset(Dataset):
    def __init__(self, hp):
        self.dataset = CorpusDataset() 
        self.hp = hp
        self.tokenizer = AutoTokenizer.from_pretrained(hp.qwen_weights, trust_remote_code=True)
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    def __len__(self): return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset.__getitem__(idx)
        sample_id, task_id, label_id, vib, instruction, response, _ = item
        
        signal_ids = SIGNAL_TOKEN_ID + encode_sample_id(sample_id)
        user_part1, user_part2 = mod_xt_for_qwen(instruction)
        
        u1_ids = self.tokenizer(user_part1, return_tensors='pt', add_special_tokens=False).input_ids[0]
        u2_ids = self.tokenizer(user_part2, return_tensors='pt', add_special_tokens=False).input_ids[0]
        
        # Construct User Input: Part1 + SignalTokens + Part2
        user_ids = torch.cat([u1_ids, signal_ids, u2_ids])
        gt_ids = self.tokenizer(response, return_tensors='pt', add_special_tokens=False).input_ids[0]
        
        input_ids = torch.cat([user_ids, gt_ids, torch.ones(1)*self.tokenizer.pad_token_id])
        attention_mask = torch.ones_like(input_ids)
        labels = torch.cat([torch.ones_like(user_ids)*-100, gt_ids, torch.ones(1)*self.tokenizer.pad_token_id])
        
        return {'input_ids': input_ids.long(), 'attention_mask': attention_mask.long(), 'labels': labels.long(), 'task_id': int(task_id)}

class MultiTaskTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        _ = inputs.pop('task_id')
        labels = inputs.get('labels', None)
        outputs = model(**{k: v for k, v in inputs.items() if k in ('input_ids', 'attention_mask', 'labels')})
        lm_loss = outputs.loss if hasattr(outputs, 'loss') and outputs.loss is not None else torch.tensor(0.0).to(inputs['input_ids'].device)
        return (lm_loss, outputs) if return_outputs else lm_loss

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=10)
    parser.add_argument('--fcn_weights', type=str, default=DEFAULT_FCN_WEIGHTS)
    parser.add_argument('--qwen_weights', type=str, default=DEFAULT_QWEN_WEIGHTS)
    parser.add_argument('--output_dir', type=str, default=DEFAULT_LORA_SAVE_DIR)
    parser.add_argument('--num_train_epochs', type=int, default=10)
    parser.add_argument('--prev_adapter', type=str, default=None)
    parser.add_argument('--resume_lora', type=str, default=None)
    parser.add_argument('--use_mecs', type=str2bool, default=True)
    
    # [Fix 4] Add mecs_type argument to CLI
    parser.add_argument('--mecs_type', type=str, default="full", 
                        choices=['full', 'b1', 'b2', 'b3', 'b4', 'b5', 'b6'])
    
    args = parser.parse_args()
    hp = HyperParameters(args)

    print(f"Start Fine-tuning | Classes: {hp.num_classes} | MECS: {hp.use_mecs} | Type: {hp.mecs_type}")

    model = get_bearllm(hp)
    
    if args.resume_lora and os.path.exists(args.resume_lora):
        print(f"Resuming LoRA from {args.resume_lora}...")
        model = PeftModel.from_pretrained(model, args.resume_lora, is_trainable=True)
    else:
        lora_config = LoraConfig(
            target_modules="all-linear",
            task_type=TaskType.CAUSAL_LM,
            r=hp.r, lora_alpha=hp.lora_alpha, lora_dropout=hp.lora_dropout
        )
        model = get_peft_model(model, lora_config)

    model.print_trainable_parameters()
    
    dataset = FineTuningDataset(hp)
    
    train_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=hp.per_device_train_batch_size,
        gradient_accumulation_steps=hp.gradient_accumulation_steps,
        logging_steps=hp.logging_steps,
        num_train_epochs=hp.num_train_epochs,
        save_steps=hp.save_steps,
        learning_rate=hp.learning_rate,
        lr_scheduler_type=hp.lr_scheduler_type,
        remove_unused_columns=False,
        fp16=True, # Enable mixed precision for speed
        report_to="none"
    )
    
    trainer = MultiTaskTrainer(model=model, args=train_args, train_dataset=dataset, data_collator=collate_fn)
    trainer.train()
    trainer.model.save_pretrained(args.output_dir)
    
    # Save Alignment Layer specifically
    adapter_save_path = os.path.join(args.output_dir, 'vibration_adapter.pth')
    
    # Need to be careful getting the state dict from wrapped model
    try:
        adapter_state = model.get_input_embeddings().adapter.alignment_layer.state_dict()
        torch.save(adapter_state, adapter_save_path)
        print(f"Adapter weights saved to {adapter_save_path}")
    except Exception as e:
        print(f"Error saving adapter separate file: {e}. (Full LoRA is already saved in output_dir)")

if __name__ == "__main__":
    main()