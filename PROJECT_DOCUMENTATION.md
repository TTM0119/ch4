# 火灾热释放速率预测系统 - 项目文档

## 项目概述

本项目基于机器学习和深度学习方法，实现火灾热释放速率的预测。通过风向、风速和时间等特征，预测火灾从起燃到熄灭过程中的热释放速率变化。

---

## 数据说明

### 原始数据文件
| 文件 | 说明 |
|------|------|
| `data.xlsx` | 原始实验数据，包含64组样本（8风向×8风速），每组4列：风向、风速、时间、热释放速率 |

### 数据特点
- **风向**: 东、南、西、北、东北、东南、西南、西北（8种）
- **风速**: 1-8 m/s
- **时间**: 从起燃到熄灭的时间序列
- **热释放速率**: 预测目标（单位：kW）

---

## 代码文件说明

### 1. 数据检查工具

#### `check_data.py`
| 项目 | 说明 |
|------|------|
| **功能** | 检查Excel数据文件结构，显示数据形状、列名、数据类型等基本信息 |
| **输入** | `data.xlsx` |
| **输出** | 控制台打印数据概览信息 |
| **意义** | 数据探索的第一步，了解原始数据结构 |

#### `quick_test.py`
| 项目 | 说明 |
|------|------|
| **功能** | 使用前4个样本快速验证数据处理和模型训练流程 |
| **输入** | `data.xlsx` |
| **输出** | 控制台打印快速测试结果（MSE、R²） |
| **意义** | 在完整训练前验证代码正确性，节省调试时间 |

---

### 2. 基础机器学习模型

#### `fire_heat_release_prediction.py`
| 项目 | 说明 |
|------|------|
| **功能** | 核心预测程序，实现多种机器学习模型（随机森林、梯度提升、线性回归）的训练、评估和比较 |
| **输入** | `data.xlsx` |
| **输出目录** | `models/`, `plots/`, `results/` |

**输出文件：**
| 文件路径 | 说明 |
|----------|------|
| `models/Random_Forest_model.pkl` | 训练好的随机森林模型 |
| `models/Gradient_Boosting_model.pkl` | 训练好的梯度提升模型 |
| `models/Linear_Regression_model.pkl` | 训练好的线性回归模型 |
| `models/label_encoder.pkl` | 风向标签编码器 |
| `plots/model_comparison.png` | 模型性能对比图（R²和RMSE） |
| `plots/scatter_plot.png` | 实际值vs预测值散点图 |
| `plots/time_series_comparison.png` | 时间序列预测对比图 |
| `plots/residual_analysis.png` | 残差分析图 |
| `results/model_performance.csv` | 模型性能指标（MSE、RMSE、MAE、R²） |

**主要结果：**
| 模型 | R² | RMSE |
|------|-----|------|
| Random Forest | 0.9608 | 2801.09 |
| Gradient Boosting | 0.8805 | 4888.77 |
| Linear Regression | 0.1653 | 12920.95 |

**意义：** 建立基准模型，验证随机森林在此任务上的优越性能。

---

### 3. 可视化脚本

#### `prediction_visualization.py`
| 项目 | 说明 |
|------|------|
| **功能** | 生成训练集和测试集的综合对比可视化 |
| **输入** | `data.xlsx` |
| **输出目录** | `visualizations/` |

**输出文件：**
| 文件路径 | 说明 |
|----------|------|
| `visualizations/train_test_comparison.png` | 训练集和测试集综合对比图（4子图） |
| `visualizations/training_set_prediction.png` | 训练集预测散点图 |
| `visualizations/test_set_prediction.png` | 测试集预测散点图 |
| `visualizations/time_series_comparison.png` | 时间序列对比图 |

**意义：** 直观展示模型在不同数据集上的预测效果。

---

#### `sample_wise_prediction.py`
| 项目 | 说明 |
|------|------|
| **功能** | 为全部64个样本分别生成预测对比图，区分训练集/测试集 |
| **输入** | `data.xlsx` |
| **输出目录** | `sample_visualizations/` |

**输出文件：**
| 文件路径 | 说明 |
|----------|------|
| `sample_visualizations/sample_01_train.png` ~ `sample_64_*.png` | 64个样本的预测对比图 |
| `sample_visualizations/summary_report.txt` | 汇总报告，包含每个样本的R²和RMSE |

**意义：** 详细分析每个样本的预测效果，发现模型在不同条件下的表现差异。

---

### 4. 数据扩增与高级模型

#### `data_augmentation.py`
| 项目 | 说明 |
|------|------|
| **功能** | 对原始数据进行4倍扩增（高斯噪声、缩放、时间偏移），对比扩增前后模型效果 |
| **输入** | `data.xlsx` |
| **输出目录** | `augmentation_results/` |

**输出文件：**
| 文件路径 | 说明 |
|----------|------|
| `augmentation_results/data_augmented.xlsx` | 扩增后的数据（256个样本） |
| `augmentation_results/augmentation_comparison.csv` | 扩增前后性能对比 |
| `augmentation_results/augmentation_comparison.png` | 可视化对比图 |
| `augmentation_results/models/best_model_augmented.pkl` | 最佳模型 |

**扩增方法：**
1. 添加高斯噪声（标准差2%）
2. 热释放速率缩放（0.97-1.03倍）
3. 时间偏移 + 噪声组合

**意义：** 通过数据扩增增加训练样本，提升模型泛化能力。

---

#### `data_augmentation_v2.py`
| 项目 | 说明 |
|------|------|
| **功能** | 改进版数据扩增，采用更先进的特征工程和模型 |
| **输入** | `data.xlsx` |
| **输出目录** | `augmentation_results_v2/` |

**改进点：**
| 改进项 | 方法 |
|--------|------|
| 风向编码 | sin/cos编码（保留环形邻近关系） |
| 时间特征 | t_norm（归一化）+ t_end（距结束时间） |
| 样本加权 | 每条曲线权重相等 |
| 目标变换 | log1p(HRR) |
| 模型 | CatBoost、LightGBM、XGBoost、RandomForest |

**输出文件：**
| 文件路径 | 说明 |
|----------|------|
| `augmentation_results_v2/data_augmented_v2.xlsx` | 扩增后的数据 |
| `augmentation_results_v2/model_performance_v2.csv` | 模型性能指标 |
| `augmentation_results_v2/model_comparison_v2.png` | 模型对比图 |
| `augmentation_results_v2/models/` | 训练好的模型 |

**主要结果：**
| 模型 | R²（原始空间） | RMSE |
|------|----------------|------|
| LightGBM | 0.9251 | 3871.48 |
| RandomForest | 0.8666 | 5164.96 |
| CatBoost | 0.8395 | 5666.10 |
| XGBoost | 0.8278 | 5868.17 |

**意义：** 验证改进特征工程（sin/cos风向编码）对模型的影响，对比不同Boosting算法。

---

### 5. 深度学习模型

#### `deep_learning_prediction.py`
| 项目 | 说明 |
|------|------|
| **功能** | 使用PyTorch实现多种深度学习模型，GPU加速训练 |
| **输入** | `data.xlsx` |
| **输出目录** | `deep_learning_results/` |

**模型架构：**
| 模型 | 结构描述 |
|------|----------|
| LSTM | 双向LSTM，2层，128隐藏单元 |
| GRU | 双向GRU，2层，128隐藏单元 |
| CNN1D | 3层1D卷积，64通道 |
| Transformer | 2层Encoder，64维，4头注意力 |
| MLP | 3层全连接（256→128→64） |

**输出文件：**
| 文件路径 | 说明 |
|----------|------|
| `deep_learning_results/LSTM_predictions.png` | LSTM测试集预测曲线 |
| `deep_learning_results/GRU_predictions.png` | GRU测试集预测曲线 |
| `deep_learning_results/CNN1D_predictions.png` | CNN1D测试集预测曲线 |
| `deep_learning_results/Transformer_predictions.png` | Transformer测试集预测曲线 |
| `deep_learning_results/dl_model_comparison.png` | 模型对比图 |
| `deep_learning_results/dl_model_performance.csv` | 性能指标 |
| `deep_learning_results/*.pth` | 模型权重文件 |

**主要结果：**
| 模型 | R² | RMSE |
|------|-----|------|
| **MLP** | **0.9398** | **3469.18** |
| Transformer | 0.9156 | 4109.15 |
| GRU | 0.8686 | 5127.13 |
| CNN1D | 0.8658 | 5181.65 |
| LSTM | 0.8656 | 5184.78 |

**意义：** 验证深度学习方法在时间序列预测任务上的效果，MLP表现最佳。

---

#### `deep_learning_optimized.py`
| 项目 | 说明 |
|------|------|
| **功能** | 优化版深度学习模型，更深网络+残差连接+Self-Attention |
| **输入** | `data.xlsx` |
| **输出目录** | `deep_learning_optimized/` |

**优化内容：**
| 优化项 | 方法 |
|--------|------|
| 训练轮数 | 100 → 200 epochs |
| 网络结构 | 更深更宽 + 残差块 |
| 注意力机制 | LSTM/GRU添加Self-Attention |
| 学习率调度 | Cosine Annealing with Warm Restarts |
| 损失函数 | MSE + MAE 混合损失 |
| 采样策略 | 曲线级等权重采样 |

**输出文件：**
| 文件路径 | 说明 |
|----------|------|
| `deep_learning_optimized/*_predictions.png` | 各模型预测曲线 |
| `deep_learning_optimized/dl_model_comparison_optimized.png` | 优化模型对比图 |
| `deep_learning_optimized/dl_optimized_performance.csv` | 性能指标 |
| `deep_learning_optimized/*.pth` | 模型权重文件 |

**主要结果：**
| 模型 | R² | RMSE |
|------|-----|------|
| MLP_Opt | 0.8859 | 4777.37 |
| CNN1D_Opt | 0.8776 | 4948.16 |
| GRU_Opt | 0.8711 | 5077.56 |
| Transformer_Opt | 0.8631 | 5232.24 |
| LSTM_Opt | 0.8475 | 5523.61 |

**意义：** 实验表明，在数据量有限的情况下，简单网络结构（原版）反而优于复杂网络（优化版）。

---

## 目录结构总览

```
2500_2/
├── data.xlsx                          # 原始数据
├── check_data.py                      # 数据检查工具
├── quick_test.py                      # 快速测试脚本
│
├── fire_heat_release_prediction.py    # 核心ML模型
├── prediction_visualization.py        # 可视化脚本
├── sample_wise_prediction.py          # 逐样本可视化
│
├── data_augmentation.py               # 数据扩增V1
├── data_augmentation_v2.py            # 数据扩增V2（改进特征）
│
├── deep_learning_prediction.py        # 深度学习模型
├── deep_learning_optimized.py         # 优化版深度学习
│
├── models/                            # 基础ML模型
├── plots/                             # 基础可视化图
├── results/                           # 基础结果
├── visualizations/                    # 训练测试对比图
├── sample_visualizations/             # 64样本预测图
├── augmentation_results/              # 数据扩增V1结果
├── augmentation_results_v2/           # 数据扩增V2结果
├── deep_learning_results/             # 深度学习结果
└── deep_learning_optimized/           # 优化DL结果
```

---

## 模型性能总排名

| 排名 | 模型 | 方法类型 | R² | RMSE (kW) |
|------|------|----------|-----|-----------|
| 1 | Random Forest (原版) | 机器学习 | 0.9608 | 2801.09 |
| 2 | MLP | 深度学习 | 0.9398 | 3469.18 |
| 3 | LightGBM | 机器学习 | 0.9251 | 3871.48 |
| 4 | Transformer | 深度学习 | 0.9156 | 4109.15 |
| 5 | Gradient Boosting | 机器学习 | 0.8805 | 4888.77 |

---

## 关键结论

1. **随机森林表现最佳**（R²=0.9608），是最推荐的模型
2. **深度学习MLP次之**（R²=0.9398），简单架构优于复杂架构
3. **特征工程很重要**：sin/cos风向编码保留环形关系
4. **数据扩增有效**：对梯度提升类模型提升明显
5. **模型复杂度需适配数据量**：小数据集上简单模型更优

---

## 运行说明

### 环境要求
- Python 3.11.5
- PyTorch 2.1.1 + CUDA 12.1
- scikit-learn, pandas, numpy, matplotlib
- catboost, lightgbm, xgboost

### 运行命令
```powershell
# 基础机器学习模型
F:\RJAZ\anaconda\python.exe fire_heat_release_prediction.py

# 数据扩增实验
F:\RJAZ\anaconda\python.exe data_augmentation_v2.py

# 深度学习模型
F:\RJAZ\anaconda\python.exe deep_learning_prediction.py
```

---

*文档生成时间：2026年3月1日*
