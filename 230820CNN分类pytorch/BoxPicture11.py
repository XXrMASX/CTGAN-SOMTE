import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
plt.rcParams["font.sans-serif"] = ["SimHei"] # 设置字体
plt.rcParams["axes.unicode_minus"] = False # 正常显示负号
#英文字体
# plt.rcParams['font.family'] = 'Times New Roman'

from sklearn.preprocessing import MinMaxScaler

# data = pd.read_csv("data/LightGBMprocess11noY.csv")  # Read the data
data = pd.read_csv("data/LightGBM333.csv")  # Read the data
scaler = MinMaxScaler()  # Create a MinMaxScaler object
normalized_data = scaler.fit_transform(data.values)  # Perform min-max normalization

plt.figure(figsize=(8, 6))
plt.boxplot(normalized_data, labels=data.columns,medianprops={'color': 'blue','linewidth': 2.5},
            flierprops={'marker': 'o', 'markerfacecolor': 'red', 'markeredgecolor': 'red'})

# plt.title("处理后数据箱型图")
#英文 轴名称
# plt.xlabel("Logging parameters",size=18)
# plt.ylabel("Normalized value",size=18)
#中文 轴名称
plt.xlabel("Logging parameters",size=22)
plt.ylabel("Normalized value",size=22)
# plt.grid(linestyle="--", alpha=0.3)

plt.xticks(rotation=90,fontsize = 22)
plt.yticks(fontsize = 22)

plt.ylim(0, 1)  # Set the y-axis limits from -0.1 to 1.1

plt.show()