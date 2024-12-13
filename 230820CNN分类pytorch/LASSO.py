import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split

# 读取CSV格式的数据文件
data = pd.read_csv('data/LightGBMprocess11.csv')

# 处理NaN值
data.fillna(data.mean(), inplace=True)

# 假设最后一列是目标变量，其余列为特征
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# 使用LASSO回归进行特征选择
lasso = Lasso(alpha=0.1)
lasso.fit(X, y)

# 输出特征系数
print("Feature coefficients:", lasso.coef_)
