 import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 读取数据集
# iris = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)
iris = pd.read_csv('data/data.csv')
# 设置数据集的列名
iris.columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class']

# 获取特征数据
X = iris.iloc[:, :-1].values

# 定义K-means算法
class KMeans:
    def __init__(self, n_clusters=3, max_iter=300):
        self.k = n_clusters  # 聚类数量
        self.max_iter = max_iter  # 最大迭代次数

    def fit(self, X):
        # 初始化聚类中心点
        self.centroids = X[np.random.choice(X.shape[0], self.k, replace=False)]
        # 迭代聚类直到收敛或达到最大迭代次数
        for i in range(self.max_iter):
            # 计算每个样本到聚类中心点的距离
            distances = np.sqrt(((X - self.centroids[:, np.newaxis])**2).sum(axis=2))
            # 分配样本到最近的聚类中心点
            self.labels = np.argmin(distances, axis=0)
            # 更新聚类中心点
            for j in range(self.k):
                self.centroids[j] = np.mean(X[self.labels == j], axis=0)

    def predict(self, X):
        # 预测新样本所属的聚类
        distances = np.sqrt(((X - self.centroids[:, np.newaxis])**2).sum(axis=2))
        return np.argmin(distances, axis=0)

# 初始化K-means聚类器
kmeans = KMeans(n_clusters=3)

# 训练模型
kmeans.fit(X)

# 预测类别
labels = kmeans.predict(X)

# 绘制聚类结果
colors = ['r', 'g', 'b']
for i in range(len(X)):
    plt.scatter(X[i][0], X[i][1], c=colors[labels[i]], s=50)
plt.scatter(kmeans.centroids[:, 0], kmeans.centroids[:, 1], c='k', marker='x', s=100)
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')
plt.show()

