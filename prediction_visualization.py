import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.pipeline import Pipeline
import os

# 设置中文字体以支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

def load_and_prepare_data(file_path):
    """
    加载并准备数据
    """
    print("正在加载数据...")
    df = pd.read_excel(file_path)
    
    # 获取每个样本的完整时间序列
    all_samples = []
    
    for i in range(1, 65):  # 1到64号样本
        wind_dir_col = f'风向_{i}'
        wind_speed_col = f'风速/m·s-1_{i}'
        time_col = f'时间/s_{i}'
        heat_rate_col = f'热释放速率/kW_{i}'
        
        # 获取当前样本的所有数据点
        sample_df = df[[wind_dir_col, wind_speed_col, time_col, heat_rate_col]].dropna()
        
        # 构建样本数据结构：每个样本包含多个时间点
        sample_data = {
            'sample_id': i,
            'wind_direction': sample_df[wind_dir_col].iloc[0],  # 样本的风向
            'wind_speed': sample_df[wind_speed_col].iloc[0],    # 样本的风速
            'time_series': []  # 时间序列数据
        }
        
        # 添加时间序列数据
        for idx, row in sample_df.iterrows():
            sample_data['time_series'].append({
                'time': row[time_col],
                'heat_rate': row[heat_rate_col]
            })
        
        all_samples.append(sample_data)
    
    print(f"总共识别了 {len(all_samples)} 个完整样本")
    
    # 获取所有可能的风向值以拟合LabelEncoder
    all_wind_directions = []
    for i in range(1, 65):  # 1到64号样本
        wind_dir_col = f'风向_{i}'
        sample_df = df[[wind_dir_col]].dropna()
        all_wind_directions.extend(sample_df[wind_dir_col].unique())
    
    # 创建并拟合LabelEncoder
    label_encoder = LabelEncoder()
    label_encoder.fit(list(set(all_wind_directions)))
    
    return all_samples, label_encoder

def prepare_sample_level_splits(all_samples, label_encoder):
    """
    按完整样本进行训练集和测试集划分
    """
    print("按完整样本进行数据划分...")
    
    # 按样本进行划分（不是按数据点）
    sample_ids = list(range(len(all_samples)))
    train_sample_ids, test_sample_ids = train_test_split(
        sample_ids, test_size=0.2, random_state=42
    )
    
    print(f"训练样本数: {len(train_sample_ids)}, 测试样本数: {len(test_sample_ids)}")
    
    # 准备训练集和测试集数据
    def extract_features_from_samples(samples, sample_indices):
        X_list = []
        y_list = []
        
        for idx in sample_indices:
            sample = samples[idx]
            wind_dir_encoded = label_encoder.transform([sample['wind_direction']])[0]
            wind_speed = sample['wind_speed']
            
            for point in sample['time_series']:
                X_list.append([wind_dir_encoded, wind_speed, point['time']])
                y_list.append(point['heat_rate'])
        
        X = np.array(X_list)
        y = np.array(y_list)
        
        return X, y
    
    X_train, y_train = extract_features_from_samples(all_samples, train_sample_ids)
    X_test, y_test = extract_features_from_samples(all_samples, test_sample_ids)
    
    return X_train, X_test, y_train, y_test, train_sample_ids, test_sample_ids

def train_best_model(X_train, y_train):
    """
    训练最佳模型（随机森林）
    """
    print("训练最佳模型（随机森林）...")
    
    model = Pipeline([
        ('scaler', StandardScaler()),
        ('regressor', RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1))
    ])
    
    model.fit(X_train, y_train)
    
    print("模型训练完成！")
    
    return model

def create_visualizations(model, X_train, X_test, y_train, y_test, train_sample_ids, test_sample_ids):
    """
    创建真实值与预测值对比图
    """
    print("生成可视化图表...")
    
    # 创建可视化目录
    os.makedirs('visualizations', exist_ok=True)
    
    # 对训练集和测试集进行预测
    print("对训练集进行预测...")
    y_train_pred = model.predict(X_train)
    
    print("对测试集进行预测...")
    y_test_pred = model.predict(X_test)
    
    # 评估模型性能
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    
    print(f"训练集 R²: {train_r2:.4f}, RMSE: {train_rmse:.2f}")
    print(f"测试集 R²: {test_r2:.4f}, RMSE: {test_rmse:.2f}")
    
    # 创建综合对比图
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. 训练集：真实值 vs 预测值
    axes[0, 0].scatter(y_train, y_train_pred, alpha=0.5, s=20, color='blue', label='训练数据点')
    axes[0, 0].plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'r--', lw=2, label='理想预测线')
    axes[0, 0].set_xlabel('真实热释放速率 (kW)')
    axes[0, 0].set_ylabel('预测热释放速率 (kW)')
    axes[0, 0].set_title(f'训练集对比\nR² = {train_r2:.4f}, RMSE = {train_rmse:.2f}')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. 测试集：真实值 vs 预测值
    axes[0, 1].scatter(y_test, y_test_pred, alpha=0.5, s=20, color='green', label='测试数据点')
    axes[0, 1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label='理想预测线')
    axes[0, 1].set_xlabel('真实热释放速率 (kW)')
    axes[0, 1].set_ylabel('预测热释放速率 (kW)')
    axes[0, 1].set_title(f'测试集对比\nR² = {test_r2:.4f}, RMSE = {test_rmse:.2f}')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. 残差图 - 训练集
    train_residuals = y_train - y_train_pred
    axes[1, 0].scatter(y_train_pred, train_residuals, alpha=0.5, s=20, color='blue')
    axes[1, 0].axhline(y=0, color='r', linestyle='--')
    axes[1, 0].set_xlabel('预测值')
    axes[1, 0].set_ylabel('残差 (真实值 - 预测值)')
    axes[1, 0].set_title('训练集残差图')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. 残差图 - 测试集
    test_residuals = y_test - y_test_pred
    axes[1, 1].scatter(y_test_pred, test_residuals, alpha=0.5, s=20, color='green')
    axes[1, 1].axhline(y=0, color='r', linestyle='--')
    axes[1, 1].set_xlabel('预测值')
    axes[1, 1].set_ylabel('残差 (真实值 - 预测值)')
    axes[1, 1].set_title('测试集残差图')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('visualizations/train_test_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 创建单独的训练集和测试集对比图
    # 训练集对比图
    plt.figure(figsize=(10, 8))
    plt.scatter(y_train, y_train_pred, alpha=0.5, s=20, color='blue', label='训练数据点')
    plt.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'r--', lw=2, label='理想预测线')
    plt.xlabel('真实热释放速率 (kW)')
    plt.ylabel('预测热释放速率 (kW)')
    plt.title(f'训练集：真实值 vs 预测值\nR² = {train_r2:.4f}, RMSE = {train_rmse:.2f}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('visualizations/training_set_prediction.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 测试集对比图
    plt.figure(figsize=(10, 8))
    plt.scatter(y_test, y_test_pred, alpha=0.5, s=20, color='green', label='测试数据点')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label='理想预测线')
    plt.xlabel('真实热释放速率 (kW)')
    plt.ylabel('预测热释放速率 (kW)')
    plt.title(f'测试集：真实值 vs 预测值\nR² = {test_r2:.4f}, RMSE = {test_rmse:.2f}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('visualizations/test_set_prediction.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 创建时间序列预测对比图（选择几个样本）
    create_time_series_comparison(model, train_sample_ids, test_sample_ids)
    
    print("可视化图表已保存到 'visualizations' 目录")

def create_time_series_comparison(model, train_sample_ids, test_sample_ids):
    """
    创建时间序列预测对比图
    """
    # 读取原始数据以获取时间序列信息
    df = pd.read_excel('data.xlsx')
    label_encoder = LabelEncoder()
    
    # 获取所有可能的风向值以拟合LabelEncoder
    all_wind_directions = []
    for i in range(1, 65):
        wind_dir_col = f'风向_{i}'
        sample_df = df[[wind_dir_col]].dropna()
        all_wind_directions.extend(sample_df[wind_dir_col].unique())
    
    label_encoder.fit(list(set(all_wind_directions)))
    
    # 选择一些训练集和测试集样本进行时间序列可视化
    selected_train_samples = train_sample_ids[:2]  # 选择前2个训练样本
    selected_test_samples = test_sample_ids[:2]    # 选择前2个测试样本
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 训练集样本时间序列
    for i, sample_idx in enumerate(selected_train_samples):
        if i >= 2:
            break
            
        wind_dir_col = f'风向_{sample_idx+1}'
        wind_speed_col = f'风速/m·s-1_{sample_idx+1}'
        time_col = f'时间/s_{sample_idx+1}'
        heat_rate_col = f'热释放速率/kW_{sample_idx+1}'
        
        sample_df = df[[wind_dir_col, wind_speed_col, time_col, heat_rate_col]].dropna()
        
        # 准备预测数据
        wind_dir_encoded = label_encoder.transform([sample_df[wind_dir_col].iloc[0]])[0]
        wind_speed = sample_df[wind_speed_col].iloc[0]
        
        X_sample = []
        for idx, row in sample_df.iterrows():
            X_sample.append([wind_dir_encoded, wind_speed, row[time_col]])
        
        X_sample = np.array(X_sample)
        y_sample_pred = model.predict(X_sample)
        y_sample_true = sample_df[heat_rate_col].values
        time_values = sample_df[time_col].values
        
        row_idx = i // 2
        col_idx = i % 2
        
        axes[row_idx, col_idx].plot(time_values, y_sample_true, label='真实值', linewidth=2, color='blue')
        axes[row_idx, col_idx].plot(time_values, y_sample_pred, label='预测值', linewidth=2, color='red', linestyle='--')
        axes[row_idx, col_idx].set_xlabel('时间 (s)')
        axes[row_idx, col_idx].set_ylabel('热释放速率 (kW)')
        axes[row_idx, col_idx].set_title(f'训练样本 {sample_idx+1} - {sample_df[wind_dir_col].iloc[0]}风, {wind_speed}m/s')
        axes[row_idx, col_idx].legend()
        axes[row_idx, col_idx].grid(True, alpha=0.3)
    
    # 测试集样本时间序列
    for i, sample_idx in enumerate(selected_test_samples):
        if i >= 2:
            break
            
        wind_dir_col = f'风向_{sample_idx+1}'
        wind_speed_col = f'风速/m·s-1_{sample_idx+1}'
        time_col = f'时间/s_{sample_idx+1}'
        heat_rate_col = f'热释放速率/kW_{sample_idx+1}'
        
        sample_df = df[[wind_dir_col, wind_speed_col, time_col, heat_rate_col]].dropna()
        
        # 准备预测数据
        wind_dir_encoded = label_encoder.transform([sample_df[wind_dir_col].iloc[0]])[0]
        wind_speed = sample_df[wind_speed_col].iloc[0]
        
        X_sample = []
        for idx, row in sample_df.iterrows():
            X_sample.append([wind_dir_encoded, wind_speed, row[time_col]])
        
        X_sample = np.array(X_sample)
        y_sample_pred = model.predict(X_sample)
        y_sample_true = sample_df[heat_rate_col].values
        time_values = sample_df[time_col].values
        
        row_idx = 1
        col_idx = i
        
        axes[row_idx, col_idx].plot(time_values, y_sample_true, label='真实值', linewidth=2, color='blue')
        axes[row_idx, col_idx].plot(time_values, y_sample_pred, label='预测值', linewidth=2, color='red', linestyle='--')
        axes[row_idx, col_idx].set_xlabel('时间 (s)')
        axes[row_idx, col_idx].set_ylabel('热释放速率 (kW)')
        axes[row_idx, col_idx].set_title(f'测试样本 {sample_idx+1} - {sample_df[wind_dir_col].iloc[0]}风, {wind_speed}m/s')
        axes[row_idx, col_idx].legend()
        axes[row_idx, col_idx].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('visualizations/time_series_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    print("="*60)
    print("火灾热释放速率预测可视化系统")
    print("="*60)
    
    # 加载和准备数据
    all_samples, label_encoder = load_and_prepare_data('data.xlsx')
    
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test, train_sample_ids, test_sample_ids = prepare_sample_level_splits(
        all_samples, label_encoder
    )
    
    print(f"训练集大小: {X_train.shape}")
    print(f"测试集大小: {X_test.shape}")
    
    # 训练最佳模型
    model = train_best_model(X_train, y_train)
    
    # 创建可视化
    create_visualizations(model, X_train, X_test, y_train, y_test, train_sample_ids, test_sample_ids)
    
    print("\n所有可视化图表已保存到 'visualizations' 目录:")
    print("- train_test_comparison.png: 训练集和测试集综合对比图")
    print("- training_set_prediction.png: 训练集预测对比图")
    print("- test_set_prediction.png: 测试集预测对比图")
    print("- time_series_comparison.png: 时间序列预测对比图")

if __name__ == "__main__":
    main()