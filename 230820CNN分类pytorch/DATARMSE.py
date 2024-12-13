import pandas as pd
import numpy as np

df = pd.read_csv('data/data -fullNOY.csv')
# df.shape
# df.info()
# pd.set_option('display.max_rows', None)  # 设置最大显示行数为 None，表示显示所有行
# df.replace(-9999, np.nan, inplace=True) #如果你的数据集中包含了-9999作为缺失值的标志，而 df.isnull().sum() 只能计算 Pandas
# 认定的缺失值（如 NaN），那么你需要先把-9999替换成 Pandas 能够识别的缺失值标志（通常是 np.nan），然后再使用 df.isnull().sum() 计算缺失值的数量
# print(df.isnull().sum())
# import numpy as np
# from scipy.interpolate import lagrange
#
# # 创建原始数据集
# data = np.array([9.253, 9.547, 9.253, 3.642, 3.642, 5.963, 8.027, 9.122, 9.824, 8.368, 3.531, 0, 0, 0, 0, 0, 0, 3.803, 3.942, 8.745])
#
# # 确定缺失值的位置
# missing_indices = [11, 12, 13, 14, 15, 16]
#
# # 使用拉格朗日插值法填补缺失值
# filled_values = lagrange(np.arange(len(data))[np.logical_not(np.isin(np.arange(len(data)), missing_indices))],
#                          data[np.logical_not(np.isin(np.arange(len(data)), missing_indices))])(missing_indices)
#
# # 填补后的数据集
# filled_data = data.copy()
# filled_data[missing_indices] = filled_values
#
# print(filled_data)
import numpy as np
from scipy import interpolate
from sklearn.metrics import mean_squared_error, mean_absolute_error
import math

# 创建原始数据集
data = np.array([9.253, 9.547, 9.253, 3.642, 3.642, 5.963, 8.027, 9.122, 9.824, 8.368, 3.531, 0, 0, 0, 0, 0, 0, 3.803, 3.942, 8.745])

# 找到缺失值的索引
missing_indices = [11, 12, 13, 14, 15, 16]

# 使用线性插值填补缺失值
non_missing_indices = np.delete(np.arange(len(data)), missing_indices)
f = interpolate.interp1d(non_missing_indices, data[non_missing_indices], kind='linear')
filled_values = f(missing_indices)

# 填补后的数据集
filled_data = data.copy()
filled_data[missing_indices] = filled_values

# 计算均方根误差(RMSE)
rmse = math.sqrt(mean_squared_error(data[non_missing_indices], filled_data[non_missing_indices]))

# 计算均方误差(MSE)
mse = mean_squared_error(data[non_missing_indices], filled_data[non_missing_indices])

# 计算平均绝对误差(MAE)
mae = mean_absolute_error(data[non_missing_indices], filled_data[non_missing_indices])

print("Filled Dataset:", filled_data)
print("RMSE:", rmse)
print("MSE:", mse)
print("MAE:", mae)