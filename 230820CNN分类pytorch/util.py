# -*- coding: utf-8 -*-
# 导入必要的库
import pandas as pd
import torch
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


# 加载数据
def data_load():
    data = pd.read_csv('data/LightGBMprocess11.csv')
    # data = pd.read_excel('data/数据集.xlsx')
    data = data.fillna(0).values  # 缺失值处理
    # data= data.fillna(data.mean(), inplace=True)
    # print(data.fillna(data.mean()))  # 用每列特征的均值填充缺失数据
    # print(data.fillna(data.median()))  # 用每列特征的中位数填充缺失数据
    # print(data.fillna(method='bfill'))  # 用相邻后面（back）特征填充前面空值
    # values = {0: 10, 1: 20, 2: 30}
    # print(data.fillna(value=values))  # 用字典对不同的列进行拉格朗日中位数填充不同的缺失数据
    # df = pd.DataFrame(data)    # 使用插值法填充缺失值
    # df_interpolated = df.interpolate()        # 使用插值法填充缺失值
    # print(df_interpolated)         # 使用插值法填充缺失值
    # print(data.max())
    # for column in df.columns:     # 使用回归方法填充缺失值
    #     X = df.loc[df[column].notnull()].index.values.reshape(-1, 1)     # 使用回归方法填充缺失值
    #     y = df.loc[df[column].notnull(), column].values.reshape(-1, 1)     # 使用回归方法填充缺失值
    #     model = LinearRegression()      # 使用回归方法填充缺失值
    #     model.fit(X, y)    # 使用回归方法填充缺失值
    #     df.loc[df[column].isnull(), column] = model.predict(df.loc[df[column].isnull()].index.values.reshape(-1, 1))
    # print(df)     # 使用回归方法填充缺失值
    # 加载数据集
    # iris = load_iris()
    X = data[:, :-1]
    y = data[:, -1]

    # 归一化效果会好一些
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # # 将标签进行独热编码
    encoder = LabelBinarizer()
    y = encoder.fit_transform(y)

    # 划分训练集和测试集
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    # 转换为 Tensor 数据类型
    # X_train = torch.tensor(X_train, dtype=torch.float32)
    # y_train = torch.tensor(y_train, dtype=torch.float32)
    # X_test = torch.tensor(X_test, dtype=torch.float32)
    # y_test = torch.tensor(y_test, dtype=torch.float32)
    # 转换为 Tensor 数据类型
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    X_val = torch.tensor(X_val, dtype=torch.float32)
    y_val = torch.tensor(y_val, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)

    # return X_train, y_train, X_test, y_test
    return X_train, y_train, X_val, y_val, X_test, y_test


# 绘制混淆矩阵
def plot_confusion(labels, predicted):
    # 绘图属性
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体
    # 真实标签和预测标签计算混淆矩阵
    cm = confusion_matrix(labels, predicted)
    # 设置类别标签
    class_names = ['0', '1', '2', '3', '4', '5']
    # 创建图表
    fig, ax = plt.subplots()
    # 绘制混淆矩阵的热力图
    sns.heatmap(cm, annot=True, cmap='Blues', fmt='d', xticklabels=class_names, yticklabels=class_names, ax=ax)
    # 添加坐标轴标签
    ax.set_xlabel('预测标签')
    ax.set_ylabel('真实标签')
    # 设置图表标题
    ax.set_title('分类预测结果混淆矩阵')
    # 自动调整布局
    plt.tight_layout()
    plt.savefig("fig/confusion.png")
    # 显示图表
    plt.show()


# 绘制散点图
def drawScatter(ds, names):
    markers = ["x", "o"]
    fig, ax = plt.subplots()
    x = range(len(ds[0]))
    for d, name, marker in zip(ds, names, markers):
        ax.scatter(x, d, alpha=0.6, label=name, marker=marker)
        ax.legend(fontsize=16, loc='upper left')
        # ax.grid(c='gray')
    plt.savefig("fig/pre.png")
    plt.show()


# 绘制损失曲线
def pltloss(train_losses):
    # 绘制损失和准确率图
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses)
    plt.title('Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.savefig("fig/Loss.png")
    plt.show()
