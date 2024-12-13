from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFECV
from sklearn.datasets import make_classification

# 生成一组样本数据
X, y = make_classification(n_samples=1000, n_features=20, n_informative=5, n_redundant=0, random_state=42)

# 定义一个随机森林分类器
clf = RandomForestClassifier(n_estimators=100, random_state=42)

# 定义REF递归特征消除器，使用5折交叉验证
rfecv = RFECV(estimator=clf, step=1, cv=5, scoring='accuracy')

# 运行REF递归特征消除器，并返回选择的特征
selected_features = rfecv.fit_transform(X, y)

# 输出选择的特征数量和选择的特征的索引
print("Selected Features: %d" % rfecv.n_features_)
print("Feature Ranking: %s" % rfecv.ranking_)

