"""
火灾热释放速率预测 - 深度学习优化版
优化点：
1. 增加训练轮数 (200 epochs)
2. 更深更宽的网络结构
3. 残差连接
4. Cosine Annealing学习率调度
5. 混合损失函数 (MSE + MAE)
6. 更好的正则化策略
7. 曲线级等权重采样
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import os
import warnings
import math
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# ===================== 风向角度映射 =====================
WIND_DIRECTION_ANGLES = {
    '东': 0, '东北': 45, '北': 90, '西北': 135,
    '西': 180, '西南': 225, '南': 270, '东南': 315
}

def wind_direction_to_sincos(direction):
    angle_deg = WIND_DIRECTION_ANGLES.get(direction, 0)
    angle_rad = np.radians(angle_deg)
    return np.sin(angle_rad), np.cos(angle_rad)


# ===================== 混合损失函数 =====================
class CombinedLoss(nn.Module):
    """MSE + MAE 混合损失"""
    def __init__(self, mse_weight=0.7, mae_weight=0.3):
        super().__init__()
        self.mse_weight = mse_weight
        self.mae_weight = mae_weight
        self.mse = nn.MSELoss()
        self.mae = nn.L1Loss()
    
    def forward(self, pred, target):
        return self.mse_weight * self.mse(pred, target) + self.mae_weight * self.mae(pred, target)


# ===================== 数据集类 =====================
class FireDataset(Dataset):
    def __init__(self, X, y, sample_ids=None):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
        self.sample_ids = sample_ids
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class FireSequenceDataset(Dataset):
    def __init__(self, sequences, targets):
        self.sequences = [torch.FloatTensor(s) for s in sequences]
        self.targets = [torch.FloatTensor(t) for t in targets]
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx]


def collate_fn(batch):
    sequences, targets = zip(*batch)
    max_len = max(s.shape[0] for s in sequences)
    
    padded_seqs = []
    padded_targets = []
    lengths = []
    
    for seq, tgt in zip(sequences, targets):
        length = seq.shape[0]
        lengths.append(length)
        
        if length < max_len:
            pad_seq = torch.zeros(max_len - length, seq.shape[1])
            padded_seqs.append(torch.cat([seq, pad_seq], dim=0))
            pad_tgt = torch.zeros(max_len - length)
            padded_targets.append(torch.cat([tgt, pad_tgt], dim=0))
        else:
            padded_seqs.append(seq)
            padded_targets.append(tgt)
    
    return torch.stack(padded_seqs), torch.stack(padded_targets), torch.LongTensor(lengths)


# ===================== 优化的深度学习模型 =====================

class ResidualBlock(nn.Module):
    """残差块"""
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim)
        )
        self.activation = nn.GELU()
    
    def forward(self, x):
        return self.activation(x + self.block(x))


class OptimizedMLP(nn.Module):
    """优化的MLP - 更深更宽 + 残差连接"""
    def __init__(self, input_dim, hidden_dim=512, num_blocks=4, dropout=0.15):
        super().__init__()
        
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        self.blocks = nn.ModuleList([
            ResidualBlock(hidden_dim) for _ in range(num_blocks)
        ])
        
        self.output = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout / 2),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, x):
        x = self.input_proj(x)
        for block in self.blocks:
            x = block(x)
        return self.output(x).squeeze(-1)


class OptimizedLSTM(nn.Module):
    """优化的LSTM - 更多层 + 残差 + Attention"""
    def __init__(self, input_dim, hidden_dim=256, num_layers=3, dropout=0.2):
        super().__init__()
        
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        self.lstm = nn.LSTM(
            hidden_dim, hidden_dim, num_layers,
            batch_first=True, dropout=dropout, bidirectional=True
        )
        
        # Self-Attention
        self.attention = nn.MultiheadAttention(hidden_dim * 2, num_heads=4, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim * 2)
        
        self.output = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout / 2),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, x, lengths=None):
        x = self.input_proj(x)
        lstm_out, _ = self.lstm(x)
        
        # Self-attention
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        out = self.norm(lstm_out + attn_out)
        
        return self.output(out).squeeze(-1)


class OptimizedGRU(nn.Module):
    """优化的GRU - 更多层 + 残差 + Attention"""
    def __init__(self, input_dim, hidden_dim=256, num_layers=3, dropout=0.2):
        super().__init__()
        
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        self.gru = nn.GRU(
            hidden_dim, hidden_dim, num_layers,
            batch_first=True, dropout=dropout, bidirectional=True
        )
        
        self.attention = nn.MultiheadAttention(hidden_dim * 2, num_heads=4, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim * 2)
        
        self.output = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout / 2),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, x, lengths=None):
        x = self.input_proj(x)
        gru_out, _ = self.gru(x)
        
        attn_out, _ = self.attention(gru_out, gru_out, gru_out)
        out = self.norm(gru_out + attn_out)
        
        return self.output(out).squeeze(-1)


class OptimizedCNN1D(nn.Module):
    """优化的1D-CNN - 残差 + 多尺度卷积"""
    def __init__(self, input_dim, hidden_channels=128, dropout=0.15):
        super().__init__()
        
        self.input_proj = nn.Conv1d(input_dim, hidden_channels, 1)
        
        # 多尺度卷积
        self.conv3 = nn.Sequential(
            nn.Conv1d(hidden_channels, hidden_channels, 3, padding=1),
            nn.BatchNorm1d(hidden_channels),
            nn.GELU()
        )
        self.conv5 = nn.Sequential(
            nn.Conv1d(hidden_channels, hidden_channels, 5, padding=2),
            nn.BatchNorm1d(hidden_channels),
            nn.GELU()
        )
        self.conv7 = nn.Sequential(
            nn.Conv1d(hidden_channels, hidden_channels, 7, padding=3),
            nn.BatchNorm1d(hidden_channels),
            nn.GELU()
        )
        
        # 深度卷积
        self.deep_conv = nn.Sequential(
            nn.Conv1d(hidden_channels * 3, hidden_channels * 2, 3, padding=1),
            nn.BatchNorm1d(hidden_channels * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_channels * 2, hidden_channels, 3, padding=1),
            nn.BatchNorm1d(hidden_channels),
            nn.GELU(),
        )
        
        self.output = nn.Sequential(
            nn.Dropout(dropout / 2),
            nn.Linear(hidden_channels, 1)
        )
    
    def forward(self, x, lengths=None):
        # x: (batch, seq_len, features) -> (batch, features, seq_len)
        x = x.permute(0, 2, 1)
        x = self.input_proj(x)
        
        # 多尺度特征
        c3 = self.conv3(x)
        c5 = self.conv5(x)
        c7 = self.conv7(x)
        
        # 拼接
        multi = torch.cat([c3, c5, c7], dim=1)
        out = self.deep_conv(multi)
        
        # (batch, channels, seq_len) -> (batch, seq_len, channels)
        out = out.permute(0, 2, 1)
        return self.output(out).squeeze(-1)


class OptimizedTransformer(nn.Module):
    """优化的Transformer - 更多层 + 位置编码改进"""
    def __init__(self, input_dim, d_model=128, nhead=8, num_layers=4, dropout=0.1):
        super().__init__()
        
        self.input_proj = nn.Linear(input_dim, d_model)
        
        # 可学习的位置编码
        self.pos_embedding = nn.Parameter(torch.randn(1, 1000, d_model) * 0.02)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=d_model * 4, dropout=dropout,
            batch_first=True, activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.norm = nn.LayerNorm(d_model)
        
        self.output = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout / 2),
            nn.Linear(d_model // 2, 1)
        )
    
    def forward(self, x, lengths=None):
        batch_size, seq_len, _ = x.shape
        x = self.input_proj(x)
        
        # 添加位置编码
        x = x + self.pos_embedding[:, :seq_len, :]
        
        x = self.transformer(x)
        x = self.norm(x)
        
        return self.output(x).squeeze(-1)


# ===================== 数据加载 =====================

def load_original_data(file_path):
    print("正在加载原始数据...")
    df = pd.read_excel(file_path)
    
    all_samples = []
    for i in range(1, 65):
        wind_dir_col = f'风向_{i}'
        wind_speed_col = f'风速/m·s-1_{i}'
        time_col = f'时间/s_{i}'
        heat_rate_col = f'热释放速率/kW_{i}'
        
        sample_df = df[[wind_dir_col, wind_speed_col, time_col, heat_rate_col]].dropna()
        
        sample_data = {
            'sample_id': i,
            'wind_direction': sample_df[wind_dir_col].iloc[0],
            'wind_speed': sample_df[wind_speed_col].iloc[0],
            'time_series': sample_df[time_col].values,
            'heat_rates': sample_df[heat_rate_col].values
        }
        all_samples.append(sample_data)
    
    print(f"成功加载 {len(all_samples)} 个原始样本")
    return all_samples


def augment_data(all_samples, augment_factor=4):
    print(f"\n数据扩增 (扩增{augment_factor}倍)...")
    np.random.seed(SEED)
    augmented_samples = []
    
    for sample in all_samples:
        augmented_samples.append({
            'sample_id': sample['sample_id'],
            'aug_id': 0,
            'wind_direction': sample['wind_direction'],
            'wind_speed': sample['wind_speed'],
            'time_series': sample['time_series'].copy(),
            'heat_rates': sample['heat_rates'].copy()
        })
        
        for aug_type in range(1, augment_factor):
            time_aug = sample['time_series'].copy()
            heat_aug = sample['heat_rates'].copy()
            
            if aug_type == 1:
                noise_std = np.std(sample['heat_rates']) * 0.02
                noise = np.random.normal(0, noise_std, len(sample['heat_rates']))
                heat_aug = sample['heat_rates'] + noise
                heat_aug = np.maximum(heat_aug, 0)
            elif aug_type == 2:
                scale_factor = np.random.uniform(0.97, 1.03)
                heat_aug = sample['heat_rates'] * scale_factor
            elif aug_type == 3:
                time_shift = np.random.uniform(-2, 2)
                time_aug = sample['time_series'] + time_shift
                time_aug = np.maximum(time_aug, 0)
                noise_std = np.std(sample['heat_rates']) * 0.015
                noise = np.random.normal(0, noise_std, len(sample['heat_rates']))
                heat_aug = sample['heat_rates'] + noise
                heat_aug = np.maximum(heat_aug, 0)
            
            augmented_samples.append({
                'sample_id': sample['sample_id'],
                'aug_id': aug_type,
                'wind_direction': sample['wind_direction'],
                'wind_speed': sample['wind_speed'],
                'time_series': time_aug,
                'heat_rates': heat_aug
            })
    
    print(f"扩增完成！样本数: {len(all_samples)} -> {len(augmented_samples)}")
    return augmented_samples


def prepare_mlp_data(samples, scaler_X=None, scaler_y=None, fit=True):
    """准备MLP数据 - 带曲线级权重"""
    X_list = []
    y_list = []
    sample_ids = []  # 用于曲线级等权重
    
    for idx, sample in enumerate(samples):
        dir_sin, dir_cos = wind_direction_to_sincos(sample['wind_direction'])
        wind_speed = sample['wind_speed']
        time_series = sample['time_series']
        heat_rates = sample['heat_rates']
        
        t_max = max(time_series.max(), 1)
        
        for t, hrr in zip(time_series, heat_rates):
            t_norm = t / t_max
            t_end = t_max - t
            X_list.append([dir_sin, dir_cos, wind_speed, t_norm, t_end])
            y_list.append(np.log1p(hrr))
            sample_ids.append(idx)
    
    X = np.array(X_list)
    y = np.array(y_list)
    sample_ids = np.array(sample_ids)
    
    if fit:
        scaler_X = StandardScaler()
        scaler_y = StandardScaler()
        X = scaler_X.fit_transform(X)
        y = scaler_y.fit_transform(y.reshape(-1, 1)).ravel()
    else:
        X = scaler_X.transform(X)
        y = scaler_y.transform(y.reshape(-1, 1)).ravel()
    
    # 计算曲线级等权重
    unique_samples = np.unique(sample_ids)
    weights = np.zeros(len(sample_ids))
    for sid in unique_samples:
        mask = sample_ids == sid
        weights[mask] = 1.0 / mask.sum()
    weights = weights / weights.sum() * len(weights)
    
    return X, y, scaler_X, scaler_y, weights, sample_ids


def prepare_sequence_data(samples, scaler=None, fit=True):
    sequences = []
    targets = []
    sample_info = []
    all_features = []
    
    for sample in samples:
        dir_sin, dir_cos = wind_direction_to_sincos(sample['wind_direction'])
        wind_speed = sample['wind_speed']
        time_series = sample['time_series']
        t_max = max(time_series.max(), 1)
        
        for t in time_series:
            t_norm = t / t_max
            t_end = t_max - t
            all_features.append([dir_sin, dir_cos, wind_speed, t_norm, t_end])
    
    all_features = np.array(all_features)
    
    if fit:
        scaler = StandardScaler()
        scaler.fit(all_features)
    
    for sample in samples:
        dir_sin, dir_cos = wind_direction_to_sincos(sample['wind_direction'])
        wind_speed = sample['wind_speed']
        time_series = sample['time_series']
        heat_rates = sample['heat_rates']
        
        t_max = max(time_series.max(), 1)
        
        seq_features = []
        for t in time_series:
            t_norm = t / t_max
            t_end = t_max - t
            seq_features.append([dir_sin, dir_cos, wind_speed, t_norm, t_end])
        
        seq_features = np.array(seq_features)
        seq_features = scaler.transform(seq_features)
        
        sequences.append(seq_features)
        targets.append(np.log1p(heat_rates))
        sample_info.append({
            'sample_id': sample['sample_id'],
            'wind_direction': sample['wind_direction'],
            'wind_speed': sample['wind_speed'],
            'time_series': time_series,
            'heat_rates': heat_rates
        })
    
    return sequences, targets, scaler, sample_info


# ===================== 训练函数 =====================

def train_mlp_optimized(model, train_loader, val_loader, device, epochs=200, lr=0.001):
    """优化的MLP训练"""
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=50, T_mult=2)
    criterion = CombinedLoss(mse_weight=0.7, mae_weight=0.3)
    
    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            pred = model(X_batch)
            loss = criterion(pred, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_loss += loss.item()
        
        scheduler.step()
        train_loss /= len(train_loader)
        
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                pred = model(X_batch)
                val_loss += criterion(pred, y_batch).item()
        
        val_loss /= len(val_loader)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
        
        if (epoch + 1) % 40 == 0:
            print(f"    Epoch {epoch+1}/{epochs}, Train: {train_loss:.6f}, Val: {val_loss:.6f}, LR: {scheduler.get_last_lr()[0]:.6f}")
        
        if patience_counter >= 30:
            print(f"    Early stopping at epoch {epoch+1}")
            break
    
    model.load_state_dict(best_model_state)
    return model


def train_sequence_optimized(model, train_loader, val_loader, device, epochs=200, lr=0.001):
    """优化的序列模型训练"""
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=50, T_mult=2)
    criterion = CombinedLoss(mse_weight=0.7, mae_weight=0.3)
    
    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for seqs, targets, lengths in train_loader:
            seqs, targets = seqs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            pred = model(seqs, lengths)
            
            mask = torch.zeros_like(targets, dtype=torch.bool)
            for i, l in enumerate(lengths):
                mask[i, :l] = True
            
            loss = criterion(pred[mask], targets[mask])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_loss += loss.item()
        
        scheduler.step()
        train_loss /= len(train_loader)
        
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for seqs, targets, lengths in val_loader:
                seqs, targets = seqs.to(device), targets.to(device)
                pred = model(seqs, lengths)
                
                mask = torch.zeros_like(targets, dtype=torch.bool)
                for i, l in enumerate(lengths):
                    mask[i, :l] = True
                
                val_loss += criterion(pred[mask], targets[mask]).item()
        
        val_loss /= len(val_loader)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
        
        if (epoch + 1) % 40 == 0:
            print(f"    Epoch {epoch+1}/{epochs}, Train: {train_loss:.6f}, Val: {val_loss:.6f}, LR: {scheduler.get_last_lr()[0]:.6f}")
        
        if patience_counter >= 30:
            print(f"    Early stopping at epoch {epoch+1}")
            break
    
    model.load_state_dict(best_model_state)
    return model


# ===================== 评估 =====================

def evaluate_mlp(model, X_test, y_test, scaler_y, device):
    model.eval()
    X_tensor = torch.FloatTensor(X_test).to(device)
    
    with torch.no_grad():
        pred_scaled = model(X_tensor).cpu().numpy()
    
    pred_log = scaler_y.inverse_transform(pred_scaled.reshape(-1, 1)).ravel()
    y_log = scaler_y.inverse_transform(y_test.reshape(-1, 1)).ravel()
    
    pred_orig = np.expm1(pred_log)
    y_orig = np.expm1(y_log)
    
    r2 = r2_score(y_orig, pred_orig)
    rmse = np.sqrt(mean_squared_error(y_orig, pred_orig))
    mae = mean_absolute_error(y_orig, pred_orig)
    
    return {'R²': r2, 'RMSE': rmse, 'MAE': mae}


def evaluate_sequence_model(model, test_samples, scaler, device):
    model.eval()
    
    all_pred = []
    all_true = []
    predictions_by_sample = []
    
    with torch.no_grad():
        for sample in test_samples:
            dir_sin, dir_cos = wind_direction_to_sincos(sample['wind_direction'])
            wind_speed = sample['wind_speed']
            time_series = sample['time_series']
            heat_rates = sample['heat_rates']
            
            t_max = max(time_series.max(), 1)
            
            seq_features = []
            for t in time_series:
                t_norm = t / t_max
                t_end = t_max - t
                seq_features.append([dir_sin, dir_cos, wind_speed, t_norm, t_end])
            
            seq_features = np.array(seq_features)
            seq_features = scaler.transform(seq_features)
            
            seq_tensor = torch.FloatTensor(seq_features).unsqueeze(0).to(device)
            pred_log = model(seq_tensor).squeeze(0).cpu().numpy()
            
            pred_orig = np.expm1(pred_log)
            
            all_pred.extend(pred_orig)
            all_true.extend(heat_rates)
            
            predictions_by_sample.append({
                'sample_id': sample['sample_id'],
                'wind_direction': sample['wind_direction'],
                'wind_speed': sample['wind_speed'],
                'time_series': time_series,
                'true_values': heat_rates,
                'pred_values': pred_orig
            })
    
    all_pred = np.array(all_pred)
    all_true = np.array(all_true)
    
    r2 = r2_score(all_true, all_pred)
    rmse = np.sqrt(mean_squared_error(all_true, all_pred))
    mae = mean_absolute_error(all_true, all_pred)
    
    return {'R²': r2, 'RMSE': rmse, 'MAE': mae}, predictions_by_sample


def plot_predictions(predictions_by_sample, model_name, output_dir):
    n_samples = len(predictions_by_sample)
    n_cols = 3
    n_rows = (n_samples + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4*n_rows))
    axes = axes.flatten() if n_samples > 1 else [axes]
    
    for idx, sample in enumerate(predictions_by_sample):
        ax = axes[idx]
        
        ax.plot(sample['time_series'], sample['true_values'], 
                label='真实值', linewidth=2, color='blue')
        ax.plot(sample['time_series'], sample['pred_values'], 
                label='预测值', linewidth=2, color='red', linestyle='--')
        
        r2 = r2_score(sample['true_values'], sample['pred_values'])
        
        ax.set_xlabel('时间 (s)')
        ax.set_ylabel('热释放速率 (kW)')
        ax.set_title(f'样本{sample["sample_id"]} - {sample["wind_direction"]}风 {sample["wind_speed"]}m/s\nR²={r2:.4f}')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    for idx in range(n_samples, len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{model_name}_predictions.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  预测对比图已保存: {model_name}_predictions.png")


def plot_model_comparison(results, output_dir):
    model_names = list(results.keys())
    r2_values = [results[m]['R²'] for m in model_names]
    rmse_values = [results[m]['RMSE'] for m in model_names]
    mae_values = [results[m]['MAE'] for m in model_names]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(model_names)))
    
    # R²
    bars = axes[0].bar(model_names, r2_values, color=colors)
    axes[0].set_ylabel('R² Score')
    axes[0].set_title('测试集 R² 对比')
    axes[0].set_ylim(0, 1.05)
    axes[0].tick_params(axis='x', rotation=45)
    for bar, val in zip(bars, r2_values):
        axes[0].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.02,
                    f'{val:.4f}', ha='center', va='bottom', fontsize=9)
    
    # RMSE
    bars = axes[1].bar(model_names, rmse_values, color=colors)
    axes[1].set_ylabel('RMSE (kW)')
    axes[1].set_title('测试集 RMSE 对比')
    axes[1].tick_params(axis='x', rotation=45)
    for bar, val in zip(bars, rmse_values):
        axes[1].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 50,
                    f'{val:.1f}', ha='center', va='bottom', fontsize=9)
    
    # MAE
    bars = axes[2].bar(model_names, mae_values, color=colors)
    axes[2].set_ylabel('MAE (kW)')
    axes[2].set_title('测试集 MAE 对比')
    axes[2].tick_params(axis='x', rotation=45)
    for bar, val in zip(bars, mae_values):
        axes[2].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 30,
                    f'{val:.1f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'dl_model_comparison_optimized.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"模型对比图已保存: dl_model_comparison_optimized.png")


# ===================== 主函数 =====================

def main():
    print("="*70)
    print("火灾热释放速率预测 - 深度学习优化版")
    print("="*70)
    print("\n优化点:")
    print("  1. 增加训练轮数 (200 epochs)")
    print("  2. 更深更宽的网络 + 残差连接")
    print("  3. Self-Attention机制")
    print("  4. Cosine Annealing学习率")
    print("  5. 混合损失函数 (MSE + MAE)")
    print("  6. 曲线级等权重采样")
    print("="*70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n使用设备: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    output_dir = 'deep_learning_optimized'
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载数据
    original_samples = load_original_data('data.xlsx')
    augmented_samples = augment_data(original_samples, augment_factor=4)
    
    # 划分数据集
    print("\n划分训练/测试集...")
    _, _, _, test_ids_orig = train_test_split(
        list(range(len(original_samples))), 
        list(range(len(original_samples))),
        test_size=0.2, random_state=SEED
    )
    test_original_ids = set(original_samples[i]['sample_id'] for i in test_ids_orig)
    
    train_samples = [s for s in augmented_samples if s['sample_id'] not in test_original_ids]
    test_samples = [s for s in original_samples if s['sample_id'] in test_original_ids]
    
    print(f"训练样本: {len(train_samples)}, 测试样本: {len(test_samples)}")
    
    results = {}
    
    # ==================== 训练序列模型 ====================
    print("\n" + "="*50)
    print("训练优化的序列模型")
    print("="*50)
    
    train_seqs, train_targets, scaler, _ = prepare_sequence_data(train_samples, fit=True)
    
    train_dataset = FireSequenceDataset(train_seqs, train_targets)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)
    
    val_seqs = train_seqs[:len(train_seqs)//5]
    val_targets = train_targets[:len(train_targets)//5]
    val_dataset = FireSequenceDataset(val_seqs, val_targets)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, collate_fn=collate_fn)
    
    input_dim = 5
    
    sequence_models = {
        'LSTM_Opt': OptimizedLSTM(input_dim, hidden_dim=256, num_layers=3),
        'GRU_Opt': OptimizedGRU(input_dim, hidden_dim=256, num_layers=3),
        'CNN1D_Opt': OptimizedCNN1D(input_dim, hidden_channels=128),
        'Transformer_Opt': OptimizedTransformer(input_dim, d_model=128, nhead=8, num_layers=4)
    }
    
    for model_name, model in sequence_models.items():
        print(f"\n训练 {model_name}...")
        model = model.to(device)
        model = train_sequence_optimized(model, train_loader, val_loader, device, epochs=200, lr=0.001)
        
        metrics, predictions = evaluate_sequence_model(model, test_samples, scaler, device)
        results[model_name] = metrics
        
        print(f"  {model_name} 测试集: R²={metrics['R²']:.4f}, RMSE={metrics['RMSE']:.2f}")
        
        plot_predictions(predictions, model_name, output_dir)
        torch.save(model.state_dict(), os.path.join(output_dir, f'{model_name}.pth'))
    
    # ==================== 训练MLP ====================
    print("\n" + "="*50)
    print("训练优化的MLP模型")
    print("="*50)
    
    X_train, y_train, scaler_X, scaler_y, weights_train, _ = prepare_mlp_data(train_samples, fit=True)
    X_test, y_test, _, _, _, _ = prepare_mlp_data(test_samples, scaler_X, scaler_y, fit=False)
    
    # 使用曲线级等权重采样
    sampler = WeightedRandomSampler(weights_train, len(weights_train), replacement=True)
    train_dataset = FireDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=256, sampler=sampler)
    
    val_size = len(X_train) // 5
    val_dataset = FireDataset(X_train[:val_size], y_train[:val_size])
    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)
    
    print("\n训练 MLP_Opt...")
    mlp_model = OptimizedMLP(input_dim=5, hidden_dim=512, num_blocks=4).to(device)
    mlp_model = train_mlp_optimized(mlp_model, train_loader, val_loader, device, epochs=200, lr=0.001)
    
    mlp_metrics = evaluate_mlp(mlp_model, X_test, y_test, scaler_y, device)
    results['MLP_Opt'] = mlp_metrics
    print(f"  MLP_Opt 测试集: R²={mlp_metrics['R²']:.4f}, RMSE={mlp_metrics['RMSE']:.2f}")
    
    torch.save(mlp_model.state_dict(), os.path.join(output_dir, 'MLP_Opt.pth'))
    
    # ==================== 结果汇总 ====================
    print("\n" + "="*70)
    print("优化后深度学习模型性能对比")
    print("="*70)
    
    print(f"\n{'模型':<18} {'R²':<12} {'RMSE (kW)':<12} {'MAE (kW)':<12}")
    print("-"*55)
    
    sorted_models = sorted(results.keys(), key=lambda x: results[x]['R²'], reverse=True)
    
    for model_name in sorted_models:
        m = results[model_name]
        print(f"{model_name:<18} {m['R²']:<12.4f} {m['RMSE']:<12.2f} {m['MAE']:<12.2f}")
    
    plot_model_comparison(results, output_dir)
    
    # 保存结果
    df_results = pd.DataFrame([
        {'模型': name, 'R²': metrics['R²'], 'RMSE': metrics['RMSE'], 'MAE': metrics['MAE']}
        for name, metrics in results.items()
    ])
    df_results = df_results.sort_values('R²', ascending=False)
    df_results.to_csv(os.path.join(output_dir, 'dl_optimized_performance.csv'), index=False, encoding='utf-8-sig')
    
    print(f"\n最佳模型: {sorted_models[0]} (R²={results[sorted_models[0]]['R²']:.4f})")
    
    print("\n" + "="*70)
    print("优化实验完成！")
    print(f"结果保存在: {output_dir}/")
    print("="*70)


if __name__ == "__main__":
    main()
