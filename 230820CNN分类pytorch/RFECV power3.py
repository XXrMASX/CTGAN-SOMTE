import pandas as pd
from sklearn.feature_selection import RFECV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

# 读取数据
data = pd.read_csv('data/Clear-classification_data4.csv')

# 假设最后一列是目标列
X = data.iloc[:, :-1]  # 特征
y = data.iloc[:, -1]   # 目标

# 如果目标列是分类字符串, 需要编码
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# 使用随机森林作为基模型
model = RandomForestClassifier()

# 初始化 RFECV
rfecv = RFECV(estimator=model, step=1, cv=StratifiedKFold(5), scoring='accuracy')

# 拟合数据
rfecv.fit(X, y)

# 输出优化后选择的特征数量
print(f"Optimal number of features: {rfecv.n_features_}")

# 输出被选中的特征的索引
selected_features = rfecv.support_
selected_feature_indices = [index for index, selected in enumerate(selected_features) if selected]
print(f"Selected feature indices: {selected_feature_indices}")

# 输出特征排名
ranking = rfecv.ranking_
print(f"Feature ranking: {ranking}")