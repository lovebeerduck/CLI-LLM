import sys
import os
import argparse
import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from datasets import get_loaders  
from F2LNet import F2LNet, StandardClassifier, CosineProtoClassifier

def str2bool(v):
    if isinstance(v, bool): return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'): return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'): return False
    else: raise argparse.ArgumentTypeError('Boolean value expected.')

class HyperParameters:
    def __init__(self, args=None):
        self.batch_size = 64
        self.num_workers = 4 
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.lr = 0.001
        self.epoch_max = 50
        self.num_classes = 10
        self.save_dir = './saved_weights'
        self.resume_weights = None
        self.class_order = [] 
        self.use_mecs = True
        self.use_cosine = True
        self.mecs_type = 'full'

        if args:
            if args.lr: self.lr = args.lr
            if args.epoch_max: self.epoch_max = args.epoch_max
            self.num_classes = args.num_classes
            self.save_dir = args.save_dir
            self.resume_weights = args.resume_weights
            self.use_mecs = args.use_mecs
            self.use_cosine = args.use_cosine
            self.mecs_type = args.mecs_type
            if args.class_order: self.class_order = [int(x) for x in args.class_order.split(',')]

class PreTrainer:
    def __init__(self, hp):
        self.hp = hp
        print(f"Init Trainer | MECS={hp.use_mecs} (Type={hp.mecs_type}) | Cosine={hp.use_cosine} | Order={hp.class_order}")
        self.train_loader, self.val_loader, self.test_loader = get_loaders(self.hp.batch_size, self.hp.num_workers)
        
        self.label_map = torch.full((100,), -1, dtype=torch.long).to(self.hp.device) 
        if self.hp.class_order:
            for local_idx, global_label in enumerate(self.hp.class_order): self.label_map[global_label] = local_idx
        else: self.label_map = torch.arange(100, device=self.hp.device)

        self.model = F2LNet(self.hp.num_classes, self.hp.use_mecs, self.hp.use_cosine, self.hp.mecs_type).to(self.hp.device)
        
        if self.hp.resume_weights and os.path.exists(self.hp.resume_weights):
            print(f"🔄 Resuming from: {self.hp.resume_weights}")
            self.model.load_weights(self.hp.resume_weights)
            if isinstance(self.model.classifier, StandardClassifier):
                curr = self.model.classifier.fc.weight.shape[0]
                if curr < self.hp.num_classes:
                    self.model.classifier.add_new_class(torch.randn(self.hp.num_classes-curr, 1024).to(self.hp.device)*0.01)
            elif isinstance(self.model.classifier, CosineProtoClassifier):
                curr = self.model.classifier.prototypes.shape[0]
                if curr < self.hp.num_classes:
                    self.model.classifier.add_new_class(torch.randn(self.hp.num_classes-curr, 1024).to(self.hp.device))
            self.model.to(self.hp.device)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.hp.lr)
        self.cls_criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
        self.best_val_acc = 0.0

    def _map_labels(self, labels): return self.label_map[labels]

    def train_epoch(self):
        self.model.train()
        if self.hp.resume_weights:
            for m in self.model.modules():
                if isinstance(m, torch.nn.BatchNorm1d): m.eval()
        total_loss, correct, total = 0, 0, 0
        pbar = tqdm(self.train_loader, desc="Training", leave=False)
        for data, label, _ in pbar:
            data, label = data.to(self.hp.device), self._map_labels(label.to(self.hp.device))
            self.optimizer.zero_grad()
            logits = self.model(data)
            loss = self.cls_criterion(logits, label)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
            correct += (logits.argmax(dim=1) == label).sum().item()
            total += label.size(0)
            pbar.set_postfix({'loss': loss.item()})
        return total_loss / len(self.train_loader), correct / total

    def eval_epoch(self, loader):
        self.model.eval()
        correct, total, loss_sum = 0, 0, 0
        with torch.no_grad():
            for data, label, _ in loader:
                data, label = data.to(self.hp.device), self._map_labels(label.to(self.hp.device))
                logits = self.model(data)
                loss_sum += self.cls_criterion(logits, label).item()
                correct += (logits.argmax(dim=1) == label).sum().item()
                total += label.size(0)
        return (loss_sum / len(loader), correct / total) if total > 0 else (0, 0)

    def train(self):
        for epoch in range(self.hp.epoch_max):
            train_loss, train_acc = self.train_epoch()
            val_loss, val_acc = self.eval_epoch(self.val_loader)
            print(f"Epoch {epoch+1}: Train L={train_loss:.4f} A={train_acc:.2%} | Val A={val_acc:.2%}")
            if val_acc >= self.best_val_acc:
                self.best_val_acc = val_acc
                self.model.save_weights(self.hp.save_dir)
        if not os.path.exists(self.hp.save_dir): self.model.save_weights(self.hp.save_dir)

    def test(self):
        try: self.model.load_weights(self.hp.save_dir)
        except: pass
        self.model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for data, label, _ in self.test_loader:
                data, label = data.to(self.hp.device), self._map_labels(label.to(self.hp.device))
                all_preds.extend(self.model(data).argmax(dim=1).cpu().numpy())
                all_labels.extend(label.cpu().numpy())
        
        acc = accuracy_score(all_labels, all_preds)
        prec = precision_score(all_labels, all_preds, average='macro', zero_division=0)
        rec = recall_score(all_labels, all_preds, average='macro', zero_division=0)
        f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
        print(f"🏆 Test: Acc={acc:.2%} | Prec={prec:.2%} | Rec={rec:.2%} | F1={f1:.2%}")
        
        with open(os.path.join(self.hp.save_dir, "cnn_metrics.json"), "w") as f:
            json.dump({"acc": acc, "prec": prec, "rec": rec, "f1": f1}, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=10)
    parser.add_argument('--save_dir', type=str, default='./saved_weights')
    parser.add_argument('--resume_weights', type=str, default=None)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--epoch_max', type=int, default=50)
    parser.add_argument('--use_mecs', type=str2bool, default=True)
    parser.add_argument('--use_cosine', type=str2bool, default=True)
    parser.add_argument('--class_order', type=str, default="")
    # [关键] 支持所有变体 b1-b5 以及 full
    parser.add_argument('--mecs_type', type=str, default="full", 
                        choices=['full', 'b1', 'b2', 'b3', 'b4', 'b5']) 
    args = parser.parse_args()
    trainer = PreTrainer(HyperParameters(args))
    trainer.train(); trainer.test()