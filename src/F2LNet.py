import torch
from torch import nn
import torch.nn.functional as F
import os

# =================================================================
# 1. MECS 及其 6 种变体 (消融实验专用)
# =================================================================

# -----------------------------------------------------------------
# [B6] Ours (Full): Multi-scale + Attn + Gating (Mult)
# -----------------------------------------------------------------
class Light_MECS(nn.Module):
    def __init__(self, in_channels):
        super(Light_MECS, self).__init__()
        self.branch1 = nn.Conv1d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels)
        self.branch2 = nn.Conv1d(in_channels, in_channels, kernel_size=5, padding=2, groups=in_channels)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(in_channels, in_channels // 2, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(in_channels // 2, in_channels, kernel_size=1),
            nn.Sigmoid()
        )
        self.fusion = nn.Conv1d(in_channels * 2, in_channels, kernel_size=1)
        nn.init.constant_(self.fusion.weight, 0); nn.init.constant_(self.fusion.bias, 0)

    def forward(self, x):
        identity = x
        out = torch.cat([self.branch1(x), self.branch2(x)], dim=1)
        out = self.fusion(out)
        att = self.se(out)
        out = out * att # Gating (Mult)
        return identity + out

# -----------------------------------------------------------------
# [B5] w/o Gating: Multi-scale + Attn + Addition (No Gate)
# -----------------------------------------------------------------
class Light_MECS_Wo_Gating(nn.Module):
    def __init__(self, in_channels):
        super(Light_MECS_Wo_Gating, self).__init__()
        self.branch1 = nn.Conv1d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels)
        self.branch2 = nn.Conv1d(in_channels, in_channels, kernel_size=5, padding=2, groups=in_channels)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(in_channels, in_channels // 2, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(in_channels // 2, in_channels, kernel_size=1),
            nn.Sigmoid()
        )
        self.fusion = nn.Conv1d(in_channels * 2, in_channels, kernel_size=1)
        nn.init.constant_(self.fusion.weight, 0); nn.init.constant_(self.fusion.bias, 0)

    def forward(self, x):
        identity = x
        out = torch.cat([self.branch1(x), self.branch2(x)], dim=1)
        out = self.fusion(out)
        att = self.se(out)
        out = out + att # Addition (Enhance only)
        return identity + out

# -----------------------------------------------------------------
# [B4] w/o Scale: Single-scale + Attn + Gating (Mult)
# -----------------------------------------------------------------
class Light_MECS_Wo_Scale(nn.Module):
    def __init__(self, in_channels):
        super(Light_MECS_Wo_Scale, self).__init__()
        # Single Scale (3x3 only)
        self.single = nn.Conv1d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(in_channels, in_channels // 2, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(in_channels // 2, in_channels, kernel_size=1),
            nn.Sigmoid()
        )
        self.fusion = nn.Conv1d(in_channels, in_channels, kernel_size=1)
        nn.init.constant_(self.fusion.weight, 0); nn.init.constant_(self.fusion.bias, 0)

    def forward(self, x):
        identity = x
        out = self.single(x)
        out = self.fusion(out)
        att = self.se(out)
        out = out * att # Gating
        return identity + out

# -----------------------------------------------------------------
# [B3] w/o Scale & Gate: Single-scale + Attn + Addition
# -----------------------------------------------------------------
class Light_MECS_Wo_Scale_Wo_Gating(nn.Module):
    def __init__(self, in_channels):
        super(Light_MECS_Wo_Scale_Wo_Gating, self).__init__()
        self.single = nn.Conv1d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(in_channels, in_channels // 2, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(in_channels // 2, in_channels, kernel_size=1),
            nn.Sigmoid()
        )
        self.fusion = nn.Conv1d(in_channels, in_channels, kernel_size=1)
        nn.init.constant_(self.fusion.weight, 0); nn.init.constant_(self.fusion.bias, 0)

    def forward(self, x):
        identity = x
        out = self.single(x)
        out = self.fusion(out)
        att = self.se(out)
        out = out + att # Addition
        return identity + out

# -----------------------------------------------------------------
# [B2] w/o Attn: Multi-scale + No Attn
# -----------------------------------------------------------------
class Light_MECS_Wo_Attn(nn.Module):
    def __init__(self, in_channels):
        super(Light_MECS_Wo_Attn, self).__init__()
        self.branch1 = nn.Conv1d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels)
        self.branch2 = nn.Conv1d(in_channels, in_channels, kernel_size=5, padding=2, groups=in_channels)
        self.fusion = nn.Conv1d(in_channels * 2, in_channels, kernel_size=1)
        nn.init.constant_(self.fusion.weight, 0); nn.init.constant_(self.fusion.bias, 0)

    def forward(self, x):
        identity = x
        out = torch.cat([self.branch1(x), self.branch2(x)], dim=1)
        out = self.fusion(out)
        return identity + out

# -----------------------------------------------------------------
# [B1] Single Scale Only: Single-scale + No Attn
# -----------------------------------------------------------------
class Light_MECS_Single(nn.Module):
    def __init__(self, in_channels):
        super(Light_MECS_Single, self).__init__()
        self.single = nn.Conv1d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels)
        self.fusion = nn.Conv1d(in_channels, in_channels, kernel_size=1)
        nn.init.constant_(self.fusion.weight, 0); nn.init.constant_(self.fusion.bias, 0)

    def forward(self, x):
        identity = x
        out = self.single(x)
        out = self.fusion(out)
        return identity + out


# =================================================================
# 2. Backbone (工厂更新)
# =================================================================
class WDCNN_Backbone(nn.Module):
    def __init__(self, in_channels=1, out_channels=1024, use_mecs=True, mecs_type='full'):
        super(WDCNN_Backbone, self).__init__()
        self.use_mecs = use_mecs
        
        # 🏭 变体工厂
        def build_block(ch):
            if mecs_type == 'b1': return Light_MECS_Single(ch)
            if mecs_type == 'b2': return Light_MECS_Wo_Attn(ch)
            if mecs_type == 'b3': return Light_MECS_Wo_Scale_Wo_Gating(ch)
            if mecs_type == 'b4': return Light_MECS_Wo_Scale(ch)
            if mecs_type == 'b5': return Light_MECS_Wo_Gating(ch)
            return Light_MECS(ch) # b6/full

        self.layer1 = nn.Sequential(nn.Conv1d(in_channels, 16, 64, 16, 24), nn.BatchNorm1d(16), nn.ReLU(), nn.MaxPool1d(2, 2))
        if self.use_mecs: self.mecs1 = build_block(16)

        self.layer2 = nn.Sequential(nn.Conv1d(16, 32, 3, 1, 1), nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(2, 2))
        if self.use_mecs: self.mecs2 = build_block(32)

        self.layer3 = nn.Sequential(nn.Conv1d(32, 64, 3, 1, 1), nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2, 2))
        if self.use_mecs: self.mecs3 = build_block(64)

        self.layer4 = nn.Sequential(nn.Conv1d(64, 64, 3, 1, 1), nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2, 2))
        if self.use_mecs: self.mecs4 = build_block(64)

        self.layer5 = nn.Sequential(nn.Conv1d(64, 64, 3, 1, 1), nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2, 2))
        if self.use_mecs: self.mecs5 = build_block(64)

        self.global_pool = nn.AdaptiveAvgPool1d(1) 
        self.dropout = nn.Dropout(p=0.5)
        self.fc = nn.Linear(64, out_channels)

    def forward(self, x):
        x = self.layer1(x)
        if self.use_mecs: x = self.mecs1(x)
        x = self.layer2(x)
        if self.use_mecs: x = self.mecs2(x)
        x = self.layer3(x)
        if self.use_mecs: x = self.mecs3(x)
        x = self.layer4(x)
        if self.use_mecs: x = self.mecs4(x)
        x = self.layer5(x)
        if self.use_mecs: x = self.mecs5(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        return self.fc(x)

# =================================================================
# 3. Classifier & Wrapper (保持不变)
# =================================================================
class CosineProtoClassifier(nn.Module):
    def __init__(self, num_classes=10, feat_dim=1024):
        super(CosineProtoClassifier, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.prototypes = nn.Parameter(torch.randn(num_classes, feat_dim)) 
        self.scale = nn.Parameter(torch.tensor(50.0))
    def forward(self, x):
        x_norm = F.normalize(x, p=2, dim=1)
        p_norm = F.normalize(self.prototypes, p=2, dim=1)
        return torch.mm(x_norm, p_norm.t()) * self.scale
    def add_new_class(self, new_prototypes):
        new_prototypes = new_prototypes.to(self.prototypes.device)
        self.prototypes.data = torch.cat([self.prototypes.data, new_prototypes], dim=0)
        self.num_classes += new_prototypes.shape[0]
    def save_weights(self, path):
        torch.save(self.state_dict(), os.path.join(path, "classifier.pth"))
    def load_weights(self, path):
        state_dict = torch.load(os.path.join(path, "classifier.pth"), map_location='cpu')
        saved_num = state_dict['prototypes'].shape[0]
        if self.prototypes.shape[0] != saved_num:
            self.num_classes = saved_num
            self.prototypes = nn.Parameter(torch.randn(saved_num, self.feat_dim).to(self.prototypes.device))
        self.load_state_dict(state_dict)

class StandardClassifier(nn.Module):
    def __init__(self, num_classes=10, feat_dim=1024):
        super(StandardClassifier, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.fc = nn.Linear(feat_dim, num_classes)
    def forward(self, x): return self.fc(x)
    def add_new_class(self, new_prototypes):
        device = self.fc.weight.device
        new_prototypes = new_prototypes.to(device)
        old_w, old_b = self.fc.weight.data, self.fc.bias.data
        new_w = torch.cat([old_w, new_prototypes], dim=0)
        new_b = torch.cat([old_b, torch.zeros(new_prototypes.shape[0]).to(device)], dim=0)
        self.num_classes += new_prototypes.shape[0]
        self.fc = nn.Linear(self.feat_dim, self.num_classes).to(device)
        self.fc.weight.data = new_w; self.fc.bias.data = new_b
    def save_weights(self, path): torch.save(self.state_dict(), os.path.join(path, "classifier.pth"))
    def load_weights(self, path):
        state_dict = torch.load(os.path.join(path, "classifier.pth"), map_location='cpu')
        saved_num = state_dict['fc.weight'].shape[0]
        if self.num_classes != saved_num:
            self.num_classes = saved_num
            self.fc = nn.Linear(self.feat_dim, saved_num).to(self.fc.weight.device)
        self.load_state_dict(state_dict)

class FeatureEncoder(nn.Module):
    def __init__(self, use_mecs=True, mecs_type='full'):
        super(FeatureEncoder, self).__init__()
        self.model = WDCNN_Backbone(in_channels=1, use_mecs=use_mecs, mecs_type=mecs_type)
    def forward(self, x): return self.model(x)
    def save_weights(self, path): torch.save(self.model.state_dict(), os.path.join(path, "feature_encoder.pth"))
    def load_weights(self, path): self.model.load_state_dict(torch.load(os.path.join(path, "feature_encoder.pth")))

class F2LNet(nn.Module):
    def __init__(self, num_classes=10, use_mecs=True, use_cosine=True, mecs_type='full'):
        super(F2LNet, self).__init__()
        self.encoder = FeatureEncoder(use_mecs=use_mecs, mecs_type=mecs_type)
        self.classifier = CosineProtoClassifier(num_classes) if use_cosine else StandardClassifier(num_classes)
    def forward(self, x): return self.classifier(self.encoder(x))
    def save_weights(self, path):
        if not os.path.exists(path): os.makedirs(path)
        self.encoder.save_weights(path); self.classifier.save_weights(path)
    def load_weights(self, path):
        self.encoder.load_weights(path); self.classifier.load_weights(path)