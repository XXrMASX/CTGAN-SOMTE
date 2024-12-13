import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFECV
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

# 读取CSV格式的数据
data = pd.read_csv('data/Clear-classification_data4.csv')

# 删除包含NaN值的行
data = data.dropna()

# 提取特征和目标变量
X = data.iloc[:, :-1]  # 特征
y = data.iloc[:, -1]  # 目标变量

# 将分类变量转换为数值形式
le = LabelEncoder()
y = le.fit_transform(y)

# 初始化随机森林分类器
estimator = RandomForestClassifier()

# 初始化RFECV
rfecv = RFECV(estimator, step=1, cv=StratifiedKFold(5), scoring='accuracy')

# 拟合数据
rfecv.fit(X, y)      # 只用这一句速度会提升
# # 拟合数据
# for train_index, test_index in cv.split(X, y):     # 手动写下n交叉验证
#     X_train, X_test = X.iloc[train_index], X.iloc[test_index] # 模型训练和测试：依次选择其中一个子集作为测试集，
#     y_train, y_test = y[train_index], y[test_index] # 其他n-1个子集作为训练集，训练模型并在测试集上进行评估。。
#     rfe.fit(X_train, y_train)    # 这个过程重复n次，确保每个子集都被用作测试集一次。
# 输出特征的排名

print("Optimal number of features : %d" % rfecv.n_features_)
print("Feature Ranking: %s" % rfecv.ranking_)