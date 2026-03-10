import pandas as pd

# 检查Excel文件结构
try:
    df = pd.read_excel('data.xlsx')
    print("数据形状:", df.shape)
    print("\n前5行数据:")
    print(df.head())
    print("\n列名:")
    print(df.columns.tolist())
    print("\n数据类型:")
    print(df.dtypes)
    print("\n基本信息:")
    print(df.info())
    print("\n统计摘要:")
    print(df.describe())
except Exception as e:
    print(f"读取文件时出错: {e}")
    
    # 尝试使用不同的引擎
    try:
        df = pd.read_excel('data.xlsx', engine='openpyxl')
        print("使用openpyxl引擎成功读取")
        print("数据形状:", df.shape)
        print("\n前5行数据:")
        print(df.head())
    except Exception as e2:
        print(f"使用openpyxl也失败: {e2}")