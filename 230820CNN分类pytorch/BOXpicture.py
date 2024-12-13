#当要对数据预处理前和预处理后的数据进行对比时，可以使用Python的matplotlib库绘制箱型图。以下是一个简单的示例代码，用于绘制预处理前和预处理后数据的箱型图：


import pandas as pd
import matplotlib.pyplot as plt

# 读取数据集
data_before = pd.read_csv('data/LightGBM111.csv')  # 读取预处理前的数据集
data_after = pd.read_csv('data/LightGBMprocess11noYY.csv')  # 读取预处理后的数据集

# 绘制箱型图
plt.figure(figsize=(10, 6))

# 绘制预处理前的箱型图
plt.subplot(1,2,1)
plt.boxplot(data_before.values)
plt.title('Boxplot of Data Before Preprocessing')
plt.xticks(range(1, len(data_before.columns) + 1), data_before.columns, rotation=45)

# 绘制预处理后的箱型图
plt.subplot(1,2,2)
plt.boxplot(data_after.values)
plt.title('Boxplot of Data After Preprocessing')
plt.xticks(range(1, len(data_after.columns) + 1), data_after.columns, rotation=45)

plt.tight_layout()
plt.show()

# 在这个示例代码中，假设预处理前的数据集保存在名为 "data_before_preprocessing.csv" 的文件中，预处理后的数据集保存在名为 "data_after_preprocessing.csv" 的文件中。
# 在代码中首先使用pandas库的read_csv函数读取数据，然后使用matplotlib库的boxplot函数分别绘制预处理前和预处理后的数据的箱型图，最后通过plt.show()进行展示。
# 在显示的图中，左边是预处理前的箱型图，右边是预处理后的箱型图。通过对比两个箱型图，可以直观地观察数据预处理前后数据分布的变化情况。