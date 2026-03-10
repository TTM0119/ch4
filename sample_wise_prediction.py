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
    
    return train_sample_ids, test_sample_ids

def train_model_for_samples(train_sample_ids, all_samples, label_encoder):
    """
    使用训练样本训练模型
    """
    print("准备训练数据...")
    
    X_train_list = []
    y_train_list = []
    
    for idx in train_sample_ids:
        sample = all_samples[idx]
        wind_dir_encoded = label_encoder.transform([sample['wind_direction']])[0]
        wind_speed = sample['wind_speed']
        
        for point in sample['time_series']:
            X_train_list.append([wind_dir_encoded, wind_speed, point['time']])
            y_train_list.append(point['heat_rate'])
    
    X_train = np.array(X_train_list)
    y_train = np.array(y_train_list)
    
    print(f"训练数据准备完成，共 {len(X_train)} 个数据点")
    
    print("训练模型...")
    model = Pipeline([
        ('scaler', StandardScaler()),
        ('regressor', RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1))
    ])
    
    model.fit(X_train, y_train)
    
    print("模型训练完成！")
    
    return model

def predict_and_visualize_for_all_samples(model, all_samples, label_encoder, train_sample_ids, test_sample_ids):
    """
    对所有样本进行预测并生成可视化
    """
    print("开始对所有样本进行预测和可视化...")
    
    # 创建可视化目录
    os.makedirs('sample_visualizations', exist_ok=True)
    
    # 预测所有样本
    all_sample_results = []
    
    for sample_idx, sample in enumerate(all_samples):
        # 准备该样本的预测数据
        wind_dir_encoded = label_encoder.transform([sample['wind_direction']])[0]
        wind_speed = sample['wind_speed']
        
        X_sample = []
        y_sample_true = []
        time_values = []
        
        for point in sample['time_series']:
            X_sample.append([wind_dir_encoded, wind_speed, point['time']])
            y_sample_true.append(point['heat_rate'])
            time_values.append(point['time'])
        
        X_sample = np.array(X_sample)
        y_sample_true = np.array(y_sample_true)
        
        # 进行预测
        y_sample_pred = model.predict(X_sample)
        
        # 计算该样本的性能指标
        sample_r2 = r2_score(y_sample_true, y_sample_pred)
        sample_rmse = np.sqrt(mean_squared_error(y_sample_true, y_sample_pred))
        
        # 记录结果
        is_training_sample = sample_idx in train_sample_ids
        all_sample_results.append({
            'sample_id': sample['sample_id'],
            'wind_direction': sample['wind_direction'],
            'wind_speed': wind_speed,
            'time_values': time_values,
            'true_values': y_sample_true,
            'pred_values': y_sample_pred,
            'r2': sample_r2,
            'rmse': sample_rmse,
            'is_training': is_training_sample
        })
        
        # 为该样本生成可视化
        plt.figure(figsize=(10, 6))
        plt.plot(time_values, y_sample_true, label='真实值', linewidth=2, color='blue')
        plt.plot(time_values, y_sample_pred, label='预测值', linewidth=2, color='red', linestyle='--')
        plt.xlabel('时间 (s)')
        plt.ylabel('热释放速率 (kW)')
        sample_type = "训练集" if is_training_sample else "测试集"
        plt.title(f'样本 {sample["sample_id"]} ({sample_type}) - {sample["wind_direction"]}风, {wind_speed}m/s\n'
                  f'R² = {sample_r2:.4f}, RMSE = {sample_rmse:.2f}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 保存该样本的图像
        filename = f'sample_{sample["sample_id"]:02d}_{"train" if is_training_sample else "test"}.png'
        filepath = os.path.join('sample_visualizations', filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"已生成样本 {sample['sample_id']} 的可视化图 ({'训练集' if is_training_sample else '测试集'})")
    
    # 生成汇总报告
    generate_summary_report(all_sample_results)
    
    print(f"所有样本的预测和可视化已完成！图像保存在 'sample_visualizations' 目录中")

def generate_summary_report(all_sample_results):
    """
    生成汇总报告
    """
    # 按训练集和测试集分类
    train_results = [r for r in all_sample_results if r['is_training']]
    test_results = [r for r in all_sample_results if not r['is_training']]
    
    # 计算平均性能
    train_avg_r2 = np.mean([r['r2'] for r in train_results]) if train_results else 0
    train_avg_rmse = np.mean([r['rmse'] for r in train_results]) if train_results else 0
    test_avg_r2 = np.mean([r['r2'] for r in test_results]) if test_results else 0
    test_avg_rmse = np.mean([r['rmse'] for r in test_results]) if test_results else 0
    
    # 保存汇总报告
    report_content = f"""样本预测汇总报告

总样本数: {len(all_sample_results)}
训练集样本数: {len(train_results)}
测试集样本数: {len(test_results)}

训练集平均性能:
- 平均 R²: {train_avg_r2:.4f}
- 平均 RMSE: {train_avg_rmse:.2f}

测试集平均性能:
- 平均 R²: {test_avg_r2:.4f}
- 平均 RMSE: {test_avg_rmse:.2f}

详细结果:
"""
    
    for result in all_sample_results:
        sample_type = "训练集" if result['is_training'] else "测试集"
        report_content += f"样本 {result['sample_id']:2d} ({sample_type}): 风向={result['wind_direction']}, 风速={result['wind_speed']}m/s, R²={result['r2']:.4f}, RMSE={result['rmse']:.2f}\n"
    
    with open('sample_visualizations/summary_report.txt', 'w', encoding='utf-8') as f:
        f.write(report_content)
    
    print("汇总报告已保存到 'sample_visualizations/summary_report.txt'")

def main():
    print("="*60)
    print("64个样本训练集测试集预测对比可视化系统")
    print("="*60)
    
    # 加载和准备数据
    all_samples, label_encoder = load_and_prepare_data('data.xlsx')
    
    # 划分训练集和测试集
    train_sample_ids, test_sample_ids = prepare_sample_level_splits(all_samples, label_encoder)
    
    # 训练模型
    model = train_model_for_samples(train_sample_ids, all_samples, label_encoder)
    
    # 预测并可视化所有样本
    predict_and_visualize_for_all_samples(model, all_samples, label_encoder, train_sample_ids, test_sample_ids)
    
    print("\n所有64个样本的预测对比可视化已完成！")
    print("详细图像保存在 'sample_visualizations' 目录中:")
    print("- 每个样本的预测对比图")
    print("- 汇总报告 (summary_report.txt)")

if __name__ == "__main__":
    main()