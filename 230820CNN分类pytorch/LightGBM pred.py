import lightgbm as lgb
import pandas as pd
import matplotlib.pyplot as plt

# 加载数据
data = pd.read_csv('data/processed_data441.csv', encoding='latin-1')

# 分离特征和目标变量
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# 将数据转换为LightGBM的数据集格式
lgb_data = lgb.Dataset(X, label=y)

# 定义模型参数
params = {
    'objective': 'regression',  # 指定回归任务
    'metric': 'mse',  # 评估指标为均方误差
}

# 训练模型
num_round = 100
bst = lgb.train(params, lgb_data, num_round)

# 绘制特征重要性排序
lgb.plot_importance(bst, max_num_features=17, importance_type='gain', figsize=(10, 6))
plt.show()