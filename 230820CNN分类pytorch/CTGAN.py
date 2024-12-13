import pandas as pd
import numpy as np
from ctgan import CTGAN
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.mixture import BayesianGaussianMixture

# 读取数据集
df = pd.read_csv('your_dataset.csv', encoding='latin-1')

# 确保所有数据列都是数值类型
for column in df.columns:
    df[column] = pd.to_numeric(df[column], errors='coerce')

# 检查数据集的基本信息
print(df.info())
print(df.describe())

# 分离特征和标签
features = df.iloc[:, :-1]
labels = df.iloc[:, -1]

# 选择小样本类别（5和6）的数据
small_classes = df[df.iloc[:, -1].isin([5, 6])]

# 检查小样本类别的数据
print(small_classes.info())
print(small_classes.describe())

# 设置 BayesianGaussianMixture 的 n_components 参数
bgm = BayesianGaussianMixture(n_components=10)

# 使用CTGAN生成小样本数据
ctgan = CTGAN()
ctgan._transformer._bgm_transformer = bgm  # 手动设置 bgm_transformer
ctgan.fit(small_classes, discrete_columns=[df.columns[-1]])

# 生成1000条新的小样本数据
new_samples = ctgan.sample(1000)

# 合并原始数据和生成的数据
augmented_df = pd.concat([df, new_samples], ignore_index=True)

# 检查增强后的数据集
print(augmented_df.info())
print(augmented_df.describe())

# 计算绝对对数平均值和标准差
log_means = np.log(augmented_df.abs().mean())
log_std_devs = np.log(augmented_df.abs().std())

# 画出绝对对数平均值和标准差图
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.bar(log_means.index, log_means.values)
plt.title('Log Mean of Absolute Values')
plt.xticks(rotation=90)

plt.subplot(1, 2, 2)
plt.bar(log_std_devs.index, log_std_devs.values)
plt.title('Log Standard Deviation of Absolute Values')
plt.xticks(rotation=90)

plt.tight_layout()
plt.show()

# 画出每个特征的累积总和
cumulative_sum = augmented_df.cumsum()

plt.figure(figsize=(12, 6))
for column in cumulative_sum.columns[:-1]:  # 排除掉标签列
    plt.plot(cumulative_sum[column], label=column)
plt.title('Cumulative Sum of Features')
plt.legend()
plt.show()

# 计算并展示变量之间的相关性
correlation_matrix = augmented_df.corr()

plt.figure(figsize=(12, 6))
sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()