"""
火灾热释放速率数据扩增与模型训练脚本
功能：
1. 对原始64组数据进行4倍扩增
2. 保存扩增后的数据
3. 使用扩增数据训练模型
4. 对比扩增前后的模型评价指标
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.pipeline import Pipeline
import joblib
import os
from datetime import datetime

# 设置中文字体以支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

class DataAugmentor:
    """数据扩增器"""
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        np.random.seed(random_state)
    
    def augment_sample(self, time_series, heat_rates, wind_speed, augment_type):
        """
        对单个样本进行扩增
        
        扩增方法：
        1. 添加高斯噪声到热释放速率
        2. 时间轴微小平移
        3. 热释放速率微小缩放
        """
        time_aug = time_series.copy()
        heat_aug = heat_rates.copy()
        
        if augment_type == 1:
            # 方法1: 添加高斯噪声 (标准差为原始值的2%)
            noise_std = np.std(heat_rates) * 0.02
            noise = np.random.normal(0, noise_std, len(heat_rates))
            heat_aug = heat_rates + noise
            heat_aug = np.maximum(heat_aug, 0)  # 确保非负
            
        elif augment_type == 2:
            # 方法2: 热释放速率微小缩放 (0.95-1.05倍)
            scale_factor = np.random.uniform(0.97, 1.03)
            heat_aug = heat_rates * scale_factor
            
        elif augment_type == 3:
            # 方法3: 时间轴微小偏移 + 噪声组合
            time_shift = np.random.uniform(-2, 2)  # 时间偏移±2秒
            time_aug = time_series + time_shift
            time_aug = np.maximum(time_aug, 0)  # 确保时间非负
            
            # 同时添加小幅噪声
            noise_std = np.std(heat_rates) * 0.015
            noise = np.random.normal(0, noise_std, len(heat_rates))
            heat_aug = heat_rates + noise
            heat_aug = np.maximum(heat_aug, 0)
            
        elif augment_type == 4:
            # 方法4: 综合扩增 - 缩放+噪声
            scale_factor = np.random.uniform(0.98, 1.02)
            heat_aug = heat_rates * scale_factor
            
            noise_std = np.std(heat_rates) * 0.01
            noise = np.random.normal(0, noise_std, len(heat_rates))
            heat_aug = heat_aug + noise
            heat_aug = np.maximum(heat_aug, 0)
        
        return time_aug, heat_aug


def load_original_data(file_path):
    """加载原始数据"""
    print("正在加载原始数据...")
    df = pd.read_excel(file_path)
    
    all_samples = []
    
    for i in range(1, 65):  # 1到64号样本
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
    """
    对数据进行扩增
    
    参数:
        all_samples: 原始样本列表
        augment_factor: 扩增倍数 (默认4倍，即原始1份+扩增3份)
    """
    print(f"\n开始数据扩增 (扩增{augment_factor}倍)...")
    
    augmentor = DataAugmentor(random_state=42)
    augmented_samples = []
    
    for sample in all_samples:
        # 保留原始样本
        augmented_samples.append({
            'sample_id': sample['sample_id'],
            'aug_id': 0,  # 0表示原始数据
            'wind_direction': sample['wind_direction'],
            'wind_speed': sample['wind_speed'],
            'time_series': sample['time_series'].copy(),
            'heat_rates': sample['heat_rates'].copy()
        })
        
        # 生成扩增样本
        for aug_type in range(1, augment_factor):  # 1,2,3 三种扩增方法
            time_aug, heat_aug = augmentor.augment_sample(
                sample['time_series'],
                sample['heat_rates'],
                sample['wind_speed'],
                aug_type
            )
            
            augmented_samples.append({
                'sample_id': sample['sample_id'],
                'aug_id': aug_type,
                'wind_direction': sample['wind_direction'],
                'wind_speed': sample['wind_speed'],
                'time_series': time_aug,
                'heat_rates': heat_aug
            })
    
    print(f"扩增完成！原始样本: {len(all_samples)}, 扩增后样本: {len(augmented_samples)}")
    return augmented_samples


def save_augmented_data(augmented_samples, output_path):
    """保存扩增后的数据到Excel文件"""
    print(f"\n保存扩增数据到 {output_path}...")
    
    # 找出最大时间序列长度
    max_len = max(len(s['time_series']) for s in augmented_samples)
    
    # 创建DataFrame
    data_dict = {}
    
    for idx, sample in enumerate(augmented_samples):
        sample_num = idx + 1
        
        # 风向列
        wind_dir_col = f'风向_{sample_num}'
        wind_dir_data = [sample['wind_direction']] * len(sample['time_series'])
        wind_dir_data.extend([np.nan] * (max_len - len(wind_dir_data)))
        data_dict[wind_dir_col] = wind_dir_data
        
        # 风速列
        wind_speed_col = f'风速/m·s-1_{sample_num}'
        wind_speed_data = [sample['wind_speed']] * len(sample['time_series'])
        wind_speed_data.extend([np.nan] * (max_len - len(wind_speed_data)))
        data_dict[wind_speed_col] = wind_speed_data
        
        # 时间列
        time_col = f'时间/s_{sample_num}'
        time_data = list(sample['time_series'])
        time_data.extend([np.nan] * (max_len - len(time_data)))
        data_dict[time_col] = time_data
        
        # 热释放速率列
        heat_col = f'热释放速率/kW_{sample_num}'
        heat_data = list(sample['heat_rates'])
        heat_data.extend([np.nan] * (max_len - len(heat_data)))
        data_dict[heat_col] = heat_data
    
    df = pd.DataFrame(data_dict)
    df.to_excel(output_path, index=False)
    print(f"扩增数据已保存，共 {len(augmented_samples)} 个样本")


def prepare_training_data(samples, label_encoder=None):
    """准备训练数据"""
    # 收集所有风向用于编码
    all_directions = list(set(s['wind_direction'] for s in samples))
    
    if label_encoder is None:
        label_encoder = LabelEncoder()
        label_encoder.fit(all_directions)
    
    X_list = []
    y_list = []
    sample_indices = []  # 记录每个数据点属于哪个样本
    
    for idx, sample in enumerate(samples):
        wind_dir_encoded = label_encoder.transform([sample['wind_direction']])[0]
        wind_speed = sample['wind_speed']
        
        for i, (time_val, heat_val) in enumerate(zip(sample['time_series'], sample['heat_rates'])):
            X_list.append([wind_dir_encoded, wind_speed, time_val])
            y_list.append(heat_val)
            sample_indices.append(idx)
    
    X = np.array(X_list)
    y = np.array(y_list)
    
    return X, y, label_encoder, sample_indices


def train_and_evaluate_models(X_train, X_test, y_train, y_test, model_suffix=""):
    """训练和评估多个模型"""
    models = {
        'Random Forest': Pipeline([
            ('scaler', StandardScaler()),
            ('regressor', RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1))
        ]),
        'Gradient Boosting': Pipeline([
            ('scaler', StandardScaler()),
            ('regressor', GradientBoostingRegressor(n_estimators=100, random_state=42))
        ]),
        'Linear Regression': Pipeline([
            ('scaler', StandardScaler()),
            ('regressor', LinearRegression())
        ])
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"  训练 {name}...")
        model.fit(X_train, y_train)
        
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        
        results[name] = {
            'model': model,
            'train': {
                'MSE': mean_squared_error(y_train, y_pred_train),
                'RMSE': np.sqrt(mean_squared_error(y_train, y_pred_train)),
                'MAE': mean_absolute_error(y_train, y_pred_train),
                'R²': r2_score(y_train, y_pred_train)
            },
            'test': {
                'MSE': mean_squared_error(y_test, y_pred_test),
                'RMSE': np.sqrt(mean_squared_error(y_test, y_pred_test)),
                'MAE': mean_absolute_error(y_test, y_pred_test),
                'R²': r2_score(y_test, y_pred_test)
            },
            'predictions': y_pred_test
        }
    
    return results


def split_by_samples(samples, test_size=0.2, random_state=42):
    """按完整样本进行训练测试集划分"""
    sample_ids = list(range(len(samples)))
    train_ids, test_ids = train_test_split(sample_ids, test_size=test_size, random_state=random_state)
    
    train_samples = [samples[i] for i in train_ids]
    test_samples = [samples[i] for i in test_ids]
    
    return train_samples, test_samples, train_ids, test_ids


def print_comparison_table(results_original, results_augmented):
    """打印对比表格"""
    print("\n" + "="*80)
    print("模型性能对比 (测试集)")
    print("="*80)
    
    print(f"\n{'模型':<20} {'指标':<8} {'扩增前':<15} {'扩增后':<15} {'提升':<15}")
    print("-"*80)
    
    for model_name in results_original.keys():
        for metric in ['R²', 'RMSE', 'MAE']:
            orig_val = results_original[model_name]['test'][metric]
            aug_val = results_augmented[model_name]['test'][metric]
            
            if metric == 'R²':
                improvement = aug_val - orig_val
                improve_str = f"+{improvement:.4f}" if improvement > 0 else f"{improvement:.4f}"
            else:
                improvement = ((orig_val - aug_val) / orig_val) * 100
                improve_str = f"{improvement:+.2f}%" if improvement != 0 else "0%"
            
            print(f"{model_name:<20} {metric:<8} {orig_val:<15.4f} {aug_val:<15.4f} {improve_str:<15}")
        print()


def save_comparison_results(results_original, results_augmented, output_dir):
    """保存对比结果"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 创建对比DataFrame
    comparison_data = []
    
    for model_name in results_original.keys():
        for data_type in ['train', 'test']:
            for metric in ['MSE', 'RMSE', 'MAE', 'R²']:
                orig_val = results_original[model_name][data_type][metric]
                aug_val = results_augmented[model_name][data_type][metric]
                
                if metric == 'R²':
                    improvement = aug_val - orig_val
                else:
                    improvement = orig_val - aug_val
                
                comparison_data.append({
                    '模型': model_name,
                    '数据集': '训练集' if data_type == 'train' else '测试集',
                    '指标': metric,
                    '扩增前': orig_val,
                    '扩增后': aug_val,
                    '提升': improvement
                })
    
    df = pd.DataFrame(comparison_data)
    csv_path = os.path.join(output_dir, 'augmentation_comparison.csv')
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"\n对比结果已保存到: {csv_path}")
    
    return df


def plot_comparison(results_original, results_augmented, output_dir):
    """绘制对比图表"""
    os.makedirs(output_dir, exist_ok=True)
    
    model_names = list(results_original.keys())
    
    # 创建对比图
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # R² 对比
    x = np.arange(len(model_names))
    width = 0.35
    
    r2_orig = [results_original[m]['test']['R²'] for m in model_names]
    r2_aug = [results_augmented[m]['test']['R²'] for m in model_names]
    
    bars1 = axes[0, 0].bar(x - width/2, r2_orig, width, label='扩增前', color='steelblue')
    bars2 = axes[0, 0].bar(x + width/2, r2_aug, width, label='扩增后', color='coral')
    axes[0, 0].set_ylabel('R² Score')
    axes[0, 0].set_title('测试集 R² 分数对比')
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(model_names, rotation=15)
    axes[0, 0].legend()
    axes[0, 0].set_ylim(0, 1.1)
    
    # 添加数值标签
    for bar, val in zip(bars1, r2_orig):
        axes[0, 0].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.02,
                        f'{val:.3f}', ha='center', va='bottom', fontsize=9)
    for bar, val in zip(bars2, r2_aug):
        axes[0, 0].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.02,
                        f'{val:.3f}', ha='center', va='bottom', fontsize=9)
    
    # RMSE 对比
    rmse_orig = [results_original[m]['test']['RMSE'] for m in model_names]
    rmse_aug = [results_augmented[m]['test']['RMSE'] for m in model_names]
    
    bars1 = axes[0, 1].bar(x - width/2, rmse_orig, width, label='扩增前', color='steelblue')
    bars2 = axes[0, 1].bar(x + width/2, rmse_aug, width, label='扩增后', color='coral')
    axes[0, 1].set_ylabel('RMSE')
    axes[0, 1].set_title('测试集 RMSE 对比')
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(model_names, rotation=15)
    axes[0, 1].legend()
    
    # MAE 对比
    mae_orig = [results_original[m]['test']['MAE'] for m in model_names]
    mae_aug = [results_augmented[m]['test']['MAE'] for m in model_names]
    
    bars1 = axes[1, 0].bar(x - width/2, mae_orig, width, label='扩增前', color='steelblue')
    bars2 = axes[1, 0].bar(x + width/2, mae_aug, width, label='扩增后', color='coral')
    axes[1, 0].set_ylabel('MAE')
    axes[1, 0].set_title('测试集 MAE 对比')
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(model_names, rotation=15)
    axes[1, 0].legend()
    
    # R² 提升百分比
    r2_improvement = [(r2_aug[i] - r2_orig[i]) for i in range(len(model_names))]
    colors = ['green' if v > 0 else 'red' for v in r2_improvement]
    bars = axes[1, 1].bar(x, r2_improvement, color=colors)
    axes[1, 1].set_ylabel('R² 提升值')
    axes[1, 1].set_title('数据扩增后 R² 提升')
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(model_names, rotation=15)
    axes[1, 1].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    
    for bar, val in zip(bars, r2_improvement):
        axes[1, 1].text(bar.get_x() + bar.get_width()/2., 
                        bar.get_height() + 0.005 if val > 0 else bar.get_height() - 0.015,
                        f'{val:+.4f}', ha='center', va='bottom' if val > 0 else 'top', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'augmentation_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"对比图已保存到: {os.path.join(output_dir, 'augmentation_comparison.png')}")


def main():
    print("="*70)
    print("火灾热释放速率预测 - 数据扩增实验")
    print("="*70)
    
    # 创建输出目录
    output_dir = 'augmentation_results'
    os.makedirs(output_dir, exist_ok=True)
    
    # ==================== 1. 加载原始数据 ====================
    original_samples = load_original_data('data.xlsx')
    
    # ==================== 2. 原始数据模型训练 ====================
    print("\n" + "="*50)
    print("阶段1: 使用原始数据训练模型")
    print("="*50)
    
    # 按样本划分训练测试集
    train_samples_orig, test_samples_orig, train_ids, test_ids = split_by_samples(original_samples)
    print(f"原始数据 - 训练样本: {len(train_samples_orig)}, 测试样本: {len(test_samples_orig)}")
    
    # 准备训练数据
    X_train_orig, y_train_orig, label_encoder, _ = prepare_training_data(train_samples_orig)
    X_test_orig, y_test_orig, _, _ = prepare_training_data(test_samples_orig, label_encoder)
    
    print(f"原始训练集: {X_train_orig.shape[0]} 数据点")
    print(f"原始测试集: {X_test_orig.shape[0]} 数据点")
    
    # 训练模型
    print("\n训练原始数据模型...")
    results_original = train_and_evaluate_models(X_train_orig, X_test_orig, y_train_orig, y_test_orig)
    
    # ==================== 3. 数据扩增 ====================
    print("\n" + "="*50)
    print("阶段2: 数据扩增")
    print("="*50)
    
    augmented_samples = augment_data(original_samples, augment_factor=4)
    
    # 保存扩增后的数据
    augmented_data_path = os.path.join(output_dir, 'data_augmented.xlsx')
    save_augmented_data(augmented_samples, augmented_data_path)
    
    # ==================== 4. 扩增数据模型训练 ====================
    print("\n" + "="*50)
    print("阶段3: 使用扩增数据训练模型")
    print("="*50)
    
    # 按样本划分（保持原始测试集不变，只扩增训练集）
    # 获取原始测试样本的ID
    test_original_ids = set(original_samples[i]['sample_id'] for i in test_ids)
    
    # 扩增样本中，只有原始样本ID不在测试集中的才作为训练
    train_samples_aug = [s for s in augmented_samples 
                         if s['sample_id'] not in test_original_ids]
    test_samples_aug = [s for s in augmented_samples 
                        if s['sample_id'] in test_original_ids and s['aug_id'] == 0]  # 测试集只用原始数据
    
    print(f"扩增数据 - 训练样本: {len(train_samples_aug)}, 测试样本: {len(test_samples_aug)}")
    
    # 准备训练数据
    X_train_aug, y_train_aug, _, _ = prepare_training_data(train_samples_aug, label_encoder)
    X_test_aug, y_test_aug, _, _ = prepare_training_data(test_samples_aug, label_encoder)
    
    print(f"扩增训练集: {X_train_aug.shape[0]} 数据点 (扩增了 {X_train_aug.shape[0]/X_train_orig.shape[0]:.1f}倍)")
    print(f"测试集: {X_test_aug.shape[0]} 数据点 (保持原始)")
    
    # 训练模型
    print("\n训练扩增数据模型...")
    results_augmented = train_and_evaluate_models(X_train_aug, X_test_aug, y_train_aug, y_test_aug)
    
    # ==================== 5. 结果对比 ====================
    print_comparison_table(results_original, results_augmented)
    
    # 保存对比结果
    save_comparison_results(results_original, results_augmented, output_dir)
    
    # 绘制对比图
    plot_comparison(results_original, results_augmented, output_dir)
    
    # 保存最佳模型
    best_model_name = max(results_augmented.keys(), key=lambda x: results_augmented[x]['test']['R²'])
    best_model = results_augmented[best_model_name]['model']
    
    models_dir = os.path.join(output_dir, 'models')
    os.makedirs(models_dir, exist_ok=True)
    
    joblib.dump(best_model, os.path.join(models_dir, f'best_model_augmented.pkl'))
    joblib.dump(label_encoder, os.path.join(models_dir, 'label_encoder.pkl'))
    
    print(f"\n最佳模型 ({best_model_name}) 已保存")
    
    print("\n" + "="*70)
    print("实验完成！")
    print(f"结果保存在: {output_dir}/")
    print("  - data_augmented.xlsx: 扩增后的数据")
    print("  - augmentation_comparison.csv: 详细对比结果")
    print("  - augmentation_comparison.png: 可视化对比图")
    print("  - models/: 训练好的模型")
    print("="*70)


if __name__ == "__main__":
    main()
