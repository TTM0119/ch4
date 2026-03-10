"""
火灾热释放速率预测 - 改进版数据扩增与模型训练
改进点：
1. 风向转sin/cos编码（保留环形邻近关系）
2. 时间特征：t_norm（归一化）+ t_end（距结束时间）
3. 样本加权：每条曲线权重相等
4. 目标变换：log1p(HRR)
5. 模型：CatBoost / LightGBM / XGBoost（带调参） + RF对比
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.pipeline import Pipeline
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# ===================== 风向角度映射 =====================
WIND_DIRECTION_ANGLES = {
    '东': 0,
    '东北': 45,
    '北': 90,
    '西北': 135,
    '西': 180,
    '西南': 225,
    '南': 270,
    '东南': 315
}

def wind_direction_to_sincos(direction):
    """将风向转换为sin/cos特征"""
    angle_deg = WIND_DIRECTION_ANGLES.get(direction, 0)
    angle_rad = np.radians(angle_deg)
    return np.sin(angle_rad), np.cos(angle_rad)


class ImprovedDataAugmentor:
    """改进的数据扩增器"""
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        np.random.seed(random_state)
    
    def augment_sample(self, time_series, heat_rates, aug_type):
        """对单个样本进行扩增"""
        time_aug = time_series.copy()
        heat_aug = heat_rates.copy()
        
        if aug_type == 1:
            # 方法1: 添加高斯噪声
            noise_std = np.std(heat_rates) * 0.02
            noise = np.random.normal(0, noise_std, len(heat_rates))
            heat_aug = heat_rates + noise
            heat_aug = np.maximum(heat_aug, 0)
            
        elif aug_type == 2:
            # 方法2: 微小缩放
            scale_factor = np.random.uniform(0.97, 1.03)
            heat_aug = heat_rates * scale_factor
            
        elif aug_type == 3:
            # 方法3: 时间偏移 + 噪声
            time_shift = np.random.uniform(-2, 2)
            time_aug = time_series + time_shift
            time_aug = np.maximum(time_aug, 0)
            noise_std = np.std(heat_rates) * 0.015
            noise = np.random.normal(0, noise_std, len(heat_rates))
            heat_aug = heat_rates + noise
            heat_aug = np.maximum(heat_aug, 0)
        
        return time_aug, heat_aug


def load_original_data(file_path):
    """加载原始数据"""
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
    """数据扩增"""
    print(f"\n开始数据扩增 (扩增{augment_factor}倍)...")
    
    augmentor = ImprovedDataAugmentor(random_state=42)
    augmented_samples = []
    
    for sample in all_samples:
        # 原始样本
        augmented_samples.append({
            'sample_id': sample['sample_id'],
            'aug_id': 0,
            'wind_direction': sample['wind_direction'],
            'wind_speed': sample['wind_speed'],
            'time_series': sample['time_series'].copy(),
            'heat_rates': sample['heat_rates'].copy()
        })
        
        # 扩增样本
        for aug_type in range(1, augment_factor):
            time_aug, heat_aug = augmentor.augment_sample(
                sample['time_series'],
                sample['heat_rates'],
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


def prepare_features_improved(samples, fit_scaler=True, scaler=None):
    """
    准备改进的特征
    特征：[dir_sin, dir_cos, wind_speed, t_norm, t_end]
    目标：log1p(HRR)
    """
    X_list = []
    y_list = []
    sample_weights = []  # 样本权重
    sample_indices = []
    
    for idx, sample in enumerate(samples):
        # 风向 -> sin/cos
        dir_sin, dir_cos = wind_direction_to_sincos(sample['wind_direction'])
        wind_speed = sample['wind_speed']
        
        time_series = sample['time_series']
        heat_rates = sample['heat_rates']
        
        # 计算时间特征
        t_max = time_series.max() if len(time_series) > 0 else 1
        t_max = max(t_max, 1)  # 避免除零
        
        n_points = len(time_series)
        weight_per_point = 1.0 / n_points if n_points > 0 else 1.0  # 每条曲线权重相等
        
        for i, (t, hrr) in enumerate(zip(time_series, heat_rates)):
            # 时间特征
            t_norm = t / t_max  # 归一化时间 [0, 1]
            t_end = t_max - t   # 距结束时间
            
            # 特征向量: [dir_sin, dir_cos, wind_speed, t_norm, t_end]
            X_list.append([dir_sin, dir_cos, wind_speed, t_norm, t_end])
            
            # 目标: log1p(HRR)
            y_list.append(np.log1p(hrr))
            
            # 样本权重（每条曲线权重相等）
            sample_weights.append(weight_per_point)
            sample_indices.append(idx)
    
    X = np.array(X_list)
    y = np.array(y_list)
    weights = np.array(sample_weights)
    
    # 标准化特征
    if fit_scaler:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
    else:
        X = scaler.transform(X)
    
    return X, y, weights, scaler, sample_indices


def get_models_with_tuning():
    """获取带调参的模型"""
    models = {}
    
    # 1. CatBoost
    try:
        from catboost import CatBoostRegressor
        models['CatBoost'] = {
            'model': CatBoostRegressor(random_state=42, verbose=0),
            'params': {
                'iterations': [100, 200, 300],
                'depth': [4, 6, 8],
                'learning_rate': [0.05, 0.1, 0.2]
            }
        }
    except ImportError:
        print("警告: CatBoost未安装，跳过")
    
    # 2. LightGBM
    try:
        from lightgbm import LGBMRegressor
        models['LightGBM'] = {
            'model': LGBMRegressor(random_state=42, verbose=-1),
            'params': {
                'n_estimators': [100, 200, 300],
                'max_depth': [4, 6, 8],
                'learning_rate': [0.05, 0.1, 0.2],
                'num_leaves': [15, 31, 63]
            }
        }
    except ImportError:
        print("警告: LightGBM未安装，跳过")
    
    # 3. XGBoost
    try:
        from xgboost import XGBRegressor
        models['XGBoost'] = {
            'model': XGBRegressor(random_state=42, verbosity=0),
            'params': {
                'n_estimators': [100, 200, 300],
                'max_depth': [4, 6, 8],
                'learning_rate': [0.05, 0.1, 0.2]
            }
        }
    except ImportError:
        print("警告: XGBoost未安装，跳过")
    
    # 4. Random Forest (基线对比)
    models['RandomForest'] = {
        'model': RandomForestRegressor(random_state=42, n_jobs=-1),
        'params': {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 15, 20, None],
            'min_samples_split': [2, 5, 10]
        }
    }
    
    return models


def train_with_tuning(X_train, y_train, weights_train, model_config, model_name, n_iter=20):
    """带调参训练模型"""
    print(f"  训练 {model_name} (随机搜索调参)...")
    
    model = model_config['model']
    params = model_config['params']
    
    # 使用随机搜索（比网格搜索快）
    search = RandomizedSearchCV(
        model,
        params,
        n_iter=min(n_iter, np.prod([len(v) for v in params.values()])),
        cv=3,
        scoring='r2',
        random_state=42,
        n_jobs=-1
    )
    
    # 训练（带权重）
    try:
        search.fit(X_train, y_train, sample_weight=weights_train)
    except TypeError:
        # 某些模型不支持sample_weight在fit中
        search.fit(X_train, y_train)
    
    best_model = search.best_estimator_
    best_params = search.best_params_
    
    print(f"    最佳参数: {best_params}")
    
    return best_model, best_params


def evaluate_model(model, X_test, y_test, model_name):
    """评估模型（在log空间和原始空间）"""
    y_pred_log = model.predict(X_test)
    
    # log空间的指标
    r2_log = r2_score(y_test, y_pred_log)
    rmse_log = np.sqrt(mean_squared_error(y_test, y_pred_log))
    
    # 转换回原始空间
    y_test_orig = np.expm1(y_test)
    y_pred_orig = np.expm1(y_pred_log)
    
    r2_orig = r2_score(y_test_orig, y_pred_orig)
    rmse_orig = np.sqrt(mean_squared_error(y_test_orig, y_pred_orig))
    mae_orig = mean_absolute_error(y_test_orig, y_pred_orig)
    
    return {
        'log_space': {'R²': r2_log, 'RMSE': rmse_log},
        'original_space': {'R²': r2_orig, 'RMSE': rmse_orig, 'MAE': mae_orig},
        'predictions_log': y_pred_log,
        'predictions_orig': y_pred_orig
    }


def split_by_samples(samples, test_size=0.2, random_state=42):
    """按完整样本划分"""
    sample_ids = list(range(len(samples)))
    train_ids, test_ids = train_test_split(sample_ids, test_size=test_size, random_state=random_state)
    
    train_samples = [samples[i] for i in train_ids]
    test_samples = [samples[i] for i in test_ids]
    
    return train_samples, test_samples, train_ids, test_ids


def save_augmented_data(augmented_samples, output_path):
    """保存扩增后的数据"""
    print(f"\n保存扩增数据到 {output_path}...")
    
    max_len = max(len(s['time_series']) for s in augmented_samples)
    data_dict = {}
    
    for idx, sample in enumerate(augmented_samples):
        sample_num = idx + 1
        
        wind_dir_col = f'风向_{sample_num}'
        wind_dir_data = [sample['wind_direction']] * len(sample['time_series'])
        wind_dir_data.extend([np.nan] * (max_len - len(wind_dir_data)))
        data_dict[wind_dir_col] = wind_dir_data
        
        wind_speed_col = f'风速/m·s-1_{sample_num}'
        wind_speed_data = [sample['wind_speed']] * len(sample['time_series'])
        wind_speed_data.extend([np.nan] * (max_len - len(wind_speed_data)))
        data_dict[wind_speed_col] = wind_speed_data
        
        time_col = f'时间/s_{sample_num}'
        time_data = list(sample['time_series'])
        time_data.extend([np.nan] * (max_len - len(time_data)))
        data_dict[time_col] = time_data
        
        heat_col = f'热释放速率/kW_{sample_num}'
        heat_data = list(sample['heat_rates'])
        heat_data.extend([np.nan] * (max_len - len(heat_data)))
        data_dict[heat_col] = heat_data
    
    df = pd.DataFrame(data_dict)
    df.to_excel(output_path, index=False)
    print(f"扩增数据已保存，共 {len(augmented_samples)} 个样本")


def plot_comparison_v2(results, output_dir):
    """绘制改进版对比图"""
    os.makedirs(output_dir, exist_ok=True)
    
    model_names = list(results.keys())
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 原始空间 R²
    r2_values = [results[m]['original_space']['R²'] for m in model_names]
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(model_names)))
    
    bars = axes[0, 0].bar(model_names, r2_values, color=colors)
    axes[0, 0].set_ylabel('R² Score')
    axes[0, 0].set_title('测试集 R² 分数（原始空间）')
    axes[0, 0].set_ylim(0, 1.05)
    axes[0, 0].tick_params(axis='x', rotation=15)
    for bar, val in zip(bars, r2_values):
        axes[0, 0].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.02,
                        f'{val:.4f}', ha='center', va='bottom', fontsize=10)
    
    # 原始空间 RMSE
    rmse_values = [results[m]['original_space']['RMSE'] for m in model_names]
    bars = axes[0, 1].bar(model_names, rmse_values, color=colors)
    axes[0, 1].set_ylabel('RMSE (kW)')
    axes[0, 1].set_title('测试集 RMSE（原始空间）')
    axes[0, 1].tick_params(axis='x', rotation=15)
    for bar, val in zip(bars, rmse_values):
        axes[0, 1].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 50,
                        f'{val:.1f}', ha='center', va='bottom', fontsize=10)
    
    # Log空间 R²
    r2_log_values = [results[m]['log_space']['R²'] for m in model_names]
    bars = axes[1, 0].bar(model_names, r2_log_values, color=colors)
    axes[1, 0].set_ylabel('R² Score')
    axes[1, 0].set_title('测试集 R² 分数（Log空间）')
    axes[1, 0].set_ylim(0, 1.05)
    axes[1, 0].tick_params(axis='x', rotation=15)
    for bar, val in zip(bars, r2_log_values):
        axes[1, 0].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.02,
                        f'{val:.4f}', ha='center', va='bottom', fontsize=10)
    
    # MAE
    mae_values = [results[m]['original_space']['MAE'] for m in model_names]
    bars = axes[1, 1].bar(model_names, mae_values, color=colors)
    axes[1, 1].set_ylabel('MAE (kW)')
    axes[1, 1].set_title('测试集 MAE（原始空间）')
    axes[1, 1].tick_params(axis='x', rotation=15)
    for bar, val in zip(bars, mae_values):
        axes[1, 1].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 30,
                        f'{val:.1f}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'model_comparison_v2.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"对比图已保存: {os.path.join(output_dir, 'model_comparison_v2.png')}")


def save_results_v2(results, best_params_dict, output_dir):
    """保存结果"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存性能指标
    data = []
    for model_name, metrics in results.items():
        data.append({
            '模型': model_name,
            'R²_原始空间': metrics['original_space']['R²'],
            'RMSE_原始空间': metrics['original_space']['RMSE'],
            'MAE_原始空间': metrics['original_space']['MAE'],
            'R²_Log空间': metrics['log_space']['R²'],
            'RMSE_Log空间': metrics['log_space']['RMSE'],
            '最佳参数': str(best_params_dict.get(model_name, {}))
        })
    
    df = pd.DataFrame(data)
    df = df.sort_values('R²_原始空间', ascending=False)
    
    csv_path = os.path.join(output_dir, 'model_performance_v2.csv')
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"性能指标已保存: {csv_path}")
    
    return df


def main():
    print("="*70)
    print("火灾热释放速率预测 - 改进版 V2")
    print("="*70)
    print("\n改进点:")
    print("  1. 风向 → sin/cos 编码（保留环形邻近关系）")
    print("  2. 时间特征：t_norm + t_end")
    print("  3. 样本加权：每条曲线权重相等")
    print("  4. 目标变换：log1p(HRR)")
    print("  5. 模型：CatBoost/LightGBM/XGBoost（带调参）+ RF")
    print("="*70)
    
    output_dir = 'augmentation_results_v2'
    os.makedirs(output_dir, exist_ok=True)
    
    # ==================== 1. 加载和扩增数据 ====================
    original_samples = load_original_data('data.xlsx')
    augmented_samples = augment_data(original_samples, augment_factor=4)
    
    # 保存扩增数据
    save_augmented_data(augmented_samples, os.path.join(output_dir, 'data_augmented_v2.xlsx'))
    
    # ==================== 2. 划分训练测试集 ====================
    print("\n" + "="*50)
    print("划分训练测试集（按完整样本）")
    print("="*50)
    
    # 获取原始测试样本ID
    _, _, _, test_ids_orig = split_by_samples(original_samples)
    test_original_ids = set(original_samples[i]['sample_id'] for i in test_ids_orig)
    
    # 训练集用扩增数据，测试集只用原始数据
    train_samples = [s for s in augmented_samples if s['sample_id'] not in test_original_ids]
    test_samples = [s for s in augmented_samples if s['sample_id'] in test_original_ids and s['aug_id'] == 0]
    
    print(f"训练样本数: {len(train_samples)}")
    print(f"测试样本数: {len(test_samples)}")
    
    # ==================== 3. 准备特征 ====================
    print("\n" + "="*50)
    print("准备改进特征")
    print("="*50)
    
    X_train, y_train, weights_train, scaler, _ = prepare_features_improved(train_samples, fit_scaler=True)
    X_test, y_test, weights_test, _, _ = prepare_features_improved(test_samples, fit_scaler=False, scaler=scaler)
    
    print(f"训练集: {X_train.shape[0]} 数据点, 特征维度: {X_train.shape[1]}")
    print(f"测试集: {X_test.shape[0]} 数据点")
    print(f"特征: [dir_sin, dir_cos, wind_speed, t_norm, t_end]")
    print(f"目标: log1p(HRR)")
    
    # ==================== 4. 训练模型 ====================
    print("\n" + "="*50)
    print("训练模型（带调参）")
    print("="*50)
    
    models_config = get_models_with_tuning()
    results = {}
    best_params_dict = {}
    trained_models = {}
    
    for model_name, config in models_config.items():
        best_model, best_params = train_with_tuning(
            X_train, y_train, weights_train, config, model_name
        )
        
        metrics = evaluate_model(best_model, X_test, y_test, model_name)
        
        results[model_name] = metrics
        best_params_dict[model_name] = best_params
        trained_models[model_name] = best_model
        
        print(f"    {model_name} 测试集 R²(原始空间): {metrics['original_space']['R²']:.4f}, "
              f"RMSE: {metrics['original_space']['RMSE']:.2f}")
    
    # ==================== 5. 结果对比 ====================
    print("\n" + "="*70)
    print("模型性能对比")
    print("="*70)
    
    print(f"\n{'模型':<15} {'R²(原始)':<12} {'RMSE(原始)':<12} {'MAE(原始)':<12} {'R²(Log)':<12}")
    print("-"*65)
    
    sorted_models = sorted(results.keys(), key=lambda x: results[x]['original_space']['R²'], reverse=True)
    
    for model_name in sorted_models:
        m = results[model_name]
        print(f"{model_name:<15} {m['original_space']['R²']:<12.4f} "
              f"{m['original_space']['RMSE']:<12.2f} {m['original_space']['MAE']:<12.2f} "
              f"{m['log_space']['R²']:<12.4f}")
    
    # 保存结果
    df_results = save_results_v2(results, best_params_dict, output_dir)
    
    # 绘制对比图
    plot_comparison_v2(results, output_dir)
    
    # 保存最佳模型
    best_model_name = sorted_models[0]
    models_dir = os.path.join(output_dir, 'models')
    os.makedirs(models_dir, exist_ok=True)
    
    joblib.dump(trained_models[best_model_name], os.path.join(models_dir, f'best_model_{best_model_name}.pkl'))
    joblib.dump(scaler, os.path.join(models_dir, 'feature_scaler.pkl'))
    
    print(f"\n最佳模型: {best_model_name} (R²={results[best_model_name]['original_space']['R²']:.4f})")
    
    print("\n" + "="*70)
    print("实验完成！")
    print(f"结果保存在: {output_dir}/")
    print("  - data_augmented_v2.xlsx: 扩增后的数据")
    print("  - model_performance_v2.csv: 模型性能对比")
    print("  - model_comparison_v2.png: 可视化对比图")
    print("  - models/: 训练好的模型")
    print("="*70)


if __name__ == "__main__":
    main()
