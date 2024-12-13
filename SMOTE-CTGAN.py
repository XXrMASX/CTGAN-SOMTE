import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report
from imblearn.over_sampling import SMOTE
from ctgan import CTGAN
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

# 加载数据集
data = pd.read_csv('./230820CNN分类pytorch/data/Clear-classification_data4.csv', encoding='latin-1')

# 分离特征和标签
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# 数据标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 分离训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

# 使用SMOTE处理不平衡数据
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

# 构建新的训练数据框架
resampled_data = np.hstack((X_resampled, y_resampled.reshape(-1, 1)))
resampled_df = pd.DataFrame(resampled_data, columns=data.columns)

# 训练CTGAN模型
ctgan = CTGAN(epochs=10000)
discrete_columns = [data.columns[-1]]  # 确保这是你的标签列
ctgan.fit(resampled_df, discrete_columns)

# 生成合成数据
synthetic_data = ctgan.sample(15000)

# 合并合成数据和真实数据
combined_data = pd.concat([resampled_df, synthetic_data], ignore_index=True)

# 将优化后的数据集保存到CSV文件
combined_data.to_csv('optimized_dataset.csv', index=False)

# 提取特征和标签
X_train_combined = combined_data.iloc[:, :-1]
y_train_combined = combined_data.iloc[:, -1]

# 训练分类模型
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train_combined, y_train_combined)

# 预测并评估模型
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='weighted')

print(f'Accuracy: {accuracy}')
print(f'F1 Score: {f1}')
print(classification_report(y_test, y_pred))

# 相关性矩阵热图
def plot_correlation_heatmap(data, title):
    corr = data.corr()
    plt.figure(figsize=(14, 10))
    sns.heatmap(corr, annot=False, cmap='coolwarm')
    plt.title(title)
    plt.show()

# 绘制真实数据的相关性矩阵热图
plot_correlation_heatmap(pd.DataFrame(X_train, columns=data.columns[:-1]), 'Correlation Matrix - Real Data')

# 绘制生成数据的相关性矩阵热图
plot_correlation_heatmap(pd.DataFrame(X_train_combined, columns=data.columns[:-1]), 'Correlation Matrix - Synthetic Data')

# PCA降维图示
def plot_pca(real_data, synthetic_data):
    pca = PCA(n_components=2)
    real_pca = pca.fit_transform(real_data)
    synthetic_pca = pca.transform(synthetic_data)

    plt.figure(figsize=(12, 6))
    plt.scatter(real_pca[:, 0], real_pca[:, 1], label='Real Data', alpha=0.6)
    plt.scatter(synthetic_pca[:, 0], synthetic_pca[:, 1], label='Synthetic Data', alpha=0.6)
    plt.legend()
    plt.title('PCA of Real and Synthetic Data')
    plt.show()

plot_pca(X_resampled, synthetic_data.iloc[:, :-1])