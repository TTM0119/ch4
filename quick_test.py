import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import os

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

print("快速测试数据加载和基本处理...")

# 加载数据
df = pd.read_excel('data.xlsx')
print(f"数据形状: {df.shape}")

# 每4列代表一个样本，只处理前几个样本以加快速度
samples = []
sample_labels = []

for i in range(1, 5):  # 只处理前4个样本进行快速测试
    wind_dir_col = f'风向_{i}'
    wind_speed_col = f'风速/m·s-1_{i}'
    time_col = f'时间/s_{i}'
    heat_rate_col = f'热释放速率/kW_{i}'
    
    # 获取当前样本的所有数据点
    sample_data = df[[wind_dir_col, wind_speed_col, time_col, heat_rate_col]].dropna()
    
    # 为每个时间点创建特征向量
    for idx, row in sample_data.iterrows():
        sample_feature = [
            row[wind_dir_col],  # 风向
            row[wind_speed_col],  # 风速
            row[time_col]  # 时间
        ]
        sample_target = row[heat_rate_col]  # 热释放速率
        
        samples.append(sample_feature)
        sample_labels.append(sample_target)

print(f"快速测试提取了 {len(samples)} 个数据点")

# 转换为numpy数组
samples = np.array(samples)
sample_labels = np.array(sample_labels)

# 编码风向
label_encoder = LabelEncoder()
wind_directions_encoded = label_encoder.fit_transform(samples[:, 0])
X = np.column_stack([
    wind_directions_encoded.astype(float),
    samples[:, 1].astype(float),  # 风速
    samples[:, 2].astype(float)   # 时间
])

y = sample_labels.astype(float)

print(f"特征矩阵X形状: {X.shape}")
print(f"目标向量y形状: {y.shape}")

# 划分训练测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"训练集大小: {X_train.shape[0]}, 测试集大小: {X_test.shape[0]}")

# 训练简单模型
model = Pipeline([
    ('scaler', StandardScaler()),
    ('regressor', RandomForestRegressor(n_estimators=20, random_state=42))
])

print("训练模型...")
model.fit(X_train, y_train)

# 预测和评估
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"快速测试结果:")
print(f"  MSE: {mse:.2f}")
print(f"  R²: {r2:.4f}")

print("快速测试完成！原始完整版程序应该可以运行。")