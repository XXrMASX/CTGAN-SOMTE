import pandas as pd
from scipy.interpolate import lagrange

# 读取CSV数据
data = pd.read_csv('data/data -fullscript1.csv')


# 缺失值处理
def fill_missing_value(data):
    for i in data.columns:
        for j in range(len(data)):
            if data[i].isnull()[j]:
                data[i][j] = lagrange(data[i][list(range(j-1, j+2))].replace(-9999, pd.NA).dropna().index, data[i][list(range(j-1, j+2))].replace(-9999, pd.NA).dropna().values)(j)
    return data

filled_data = fill_missing_value(data)

# 异常值处理
def fill_outlier(data):
    for i in data.columns:
        median = data[i].median()
        std = data[i].std()
        for j in range(len(data)):
            # if (data[i][j] > median + 3*std) or (data[i][j] < median - 3*std):
            #if data[i][j] < -4000 or (data[i][j] > median + 3 * std) or (data[i][j] < median - 3 * std):  # 正确数据也可能会被处理
            if data[i][j] < -4000:
                data[i][j] = median
    return data

processed_data = fill_outlier(filled_data)

# 保存处理后的数据
# processed_data.to_csv('D:/230820CNN分类pytorch/data/processed_data.csv', index=False)
# 保存处理后的数据到指定路径
processed_data.to_csv('D:/230820CNN分类pytorch/230820CNN分类pytorch/data/processed_data6.csv', index=False)
