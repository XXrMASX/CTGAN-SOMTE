import lightgbm as lgb
import pandas as pd
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split


data = pd.read_csv('data/data -fact2.csv')
# 定义特征和标签
X = data.drop('target', axis=1)
y = data['target']
# 数据集划分
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 定义Lightgbm模型和训练参数
lgb_model = lgb.LGBMClassifier(boosting_type='gbdt',max_depth=5,learning_rate=0.01,n_estimators=500,objective='binary',random_state=42)

# 训练模型并输出特征重要性
lgb_model.fit(X_train,y_train)
lgb.plot_importance(lgb_model,figsize=(10,10))
print(lgb.plot_importance)

# 选择SelectFromModel特征选择方法
sfm = SelectFromModel(lgb_model,threshold='median')
sfm.fit(X_train,y_train)
# print('111')
# print('111')
# print('111')
X_train_sfm = sfm.transform(X_train)
X_test_sfm =sfm.transform(X_test)
print(X_test_sfm)
print(X_train_sfm)

# 新的特征集预测
lgb_model_sfm = lgb.LGBMClassifier(boosting_type='gbdt',max_depth=5,learning_rate=0.01,n_estimators=500,objective='binary',random_state=42)
lgb_model_sfm.fit(X_train_sfm,y_train)
y_pred = lgb_model_sfm.predict(X_test_sfm)