from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE
from sklearn.datasets import make_classification

# 生成样本数据，包含100个特征和一个二分类目标变量
X, y = make_classification(n_samples=1000, n_features=100, n_informative=20, n_redundant=0, random_state=1)

# 创建逻辑回归模型
model = LogisticRegression()

# 创建特征递归消除法对象，选择最优的20个特征
rfe = RFE(model, n_features_to_select=20)

# 使用特征递归消除法来训练模型并选择最优的20个特征
X_selected = rfe.fit_transform(X, y)

# 打印最优的20个特征的索引和名称
print(rfe.get_support(indices=True))

# 打印特征选择后的数据集
print(X_selected)