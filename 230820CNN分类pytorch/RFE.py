import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE, RFECV
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer  # 用于处理NaN值
from sklearn.model_selection import train_test_split

# from sklearn.linear_model import LogisticRegression
# from sklearn.feature_selection import RFE
# import pandas as pd
# from sklearn.datasets import make_classification
#
# # 生成样本数据，包含100个特征和一个二分类目标变量
# from sklearn.model_selection import train_test_split
#
# data = pd.read_csv('data/LightGBMprocess11.csv')
# X = data[:, :-1]
# y = data[:, -1]
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# # X, y = make_classification(n_samples=1000, n_features=100, n_informative=20, n_redundant=0, random_state=1)
#
# # 创建逻辑回归模型
# model = LogisticRegression()
#
# # 创建特征递归消除法对象，选择最优的20个特征
# rfe = RFE(model, n_features_to_select=20)
#
# # 使用特征递归消除法来训练模型并选择最优的20个特征
# X_selected = rfe.fit_transform(X, y)
#
# # 打印最优的20个特征的索引和名称
# print(rfe.get_support(indices=True))
#
# # 打印特征选择后的数据集
# print(X_selected)


# 读取CSV格式的数据文件
data = pd.read_csv('data/processed_data4.csv', encoding='latin-1')

# 处理NaN值
imp = SimpleImputer(missing_values=float("NaN"), strategy='mean')
data = imp.fit_transform(data)

# 假设最后一列是目标变量，其余列为特征
X = data[:, :-1]
y = data[:, -1]

# 初始化线性回归模型
model = LinearRegression()

# # 定义一个随机森林分类器
# clf = RandomForestClassifier(n_estimators=100, random_state=42)

# 初始化RFE对象，选择要保留的特征数量
rfe = RFE(model, n_features_to_select=3)  # 假设选择3个最重要的特征

# # 定义REF递归特征消除器，使用5折交叉验证
# rfecv = RFECV(estimator=clf, step=1, cv=5, scoring='accuracy')

# # 使用RFECV进行特征选择
# selected_features = rfecv.fit_transform(X, y)

 # 使用RFECV进行特征选择
selected_features = rfe.fit_transform(X, y)
# 输出最终选择的特征
print("Selected features:", selected_features)

# # 输出选择的特征数量和选择的特征的索引
# print("Selected Features: %d" % rfecv.n_features_)
# print("Feature Ranking: %s" % rfecv.ranking_)