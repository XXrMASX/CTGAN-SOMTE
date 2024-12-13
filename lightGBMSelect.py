import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error

data = pd.read_csv('./230820CNN分类pytorch/data/processed_data4.csv', encoding='latin-1')

np.random.seed(42)
X = data.iloc[:15496, :-1]
y = data.iloc[:15496, -1]-1


# 划分数据集
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

# 转换数据为LightGBM的数据格式
train_data = lgb.Dataset(X_train, label=y_train)
valid_data = lgb.Dataset(X_valid, label=y_valid, reference=train_data)

# 定义参数
params = {
    'objective': 'multiclass',  # 指定多分类任务
    'metric': 'multi_error',  # 指定评估指标为多分类错误率
    'num_class': 6,  # 指定类别数量
    'boosting_type': 'gbdt',  # 提升方式
    'num_leaves': 31,  # 叶子数
    'learning_rate': 0.05,     # 学习率
    'feature_fraction': 0.9     # 特征采样比例
}

# 训练模型
num_round = 100
bst = lgb.train(params, train_data, num_round, valid_sets=[valid_data], early_stopping_rounds=10)

# 预测
y_pred = np.argmax(bst.predict(X_valid), axis=1)

# 评估模型
accuracy = accuracy_score(y_valid, y_pred)
conf_matrix = confusion_matrix(y_valid, y_pred)
conf_matrix = np.round(conf_matrix / np.sum(conf_matrix, axis=1), decimals=2)
print(f'Accuracy: {accuracy}')
print('Confusion Matrix:')
print(conf_matrix)

plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='f', cmap='Blues', annot_kws={"size": 16})
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()
# 绘制特征重要性排序图
fig, ax = plt.subplots(figsize=(20, 20))
lgb.plot_importance(bst, max_num_features=75, importance_type='split', ax=ax)
plt.show()



# # 进行预测
# y_pred = bst.predict(X_valid)
# y_pred_class = np.argmax(y_pred, axis=1)  # 将概率预测值转换成类别
#
# # 计算均方根误差（RMSE）
# rmse = np.sqrt(mean_squared_error(y_valid, y_pred_class))
#
# # 计算均方误差（MSE）
# mse = mean_squared_error(y_valid, y_pred_class)
#
# # 计算平均绝对误差（MAE）
# mae = mean_absolute_error(y_valid, y_pred_class)
#
# print(f'RMSE: {rmse}')
# print(f'MSE: {mse}')
# print(f'MAE: {mae}')




