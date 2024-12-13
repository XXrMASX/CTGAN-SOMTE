import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFECV
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

# 读取CSV格式的数据文件
data = pd.read_csv('data/data -fact2.csv')

# 假设最后一列是目标变量，其余列为特征
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# 初始化随机森林分类器作为基础模型
rfc = RandomForestClassifier()

# 初始化RFECV对象，指定基础模型和交叉验证策略
rfecv = RFECV(estimator=rfc, step=1, cv=StratifiedKFold(5), scoring='accuracy')

# 使用RFECV进行特征选择
selected_features = rfecv.fit_transform(X, y)

# 输出最终选择的特征
print("Optimal number of features : %d" % rfecv.n_features_)
# print("Selected features:", selected_features)
print("Feature Ranking: %s" % rfecv.ranking_)