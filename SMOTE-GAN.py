import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE
from keras.models import Sequential
from keras.layers import Dense, BatchNormalization, LeakyReLU
from keras.optimizers import Adam
from sklearn.decomposition import PCA

# 加载数据集
data = pd.read_csv('./230820CNN分类pytorch/data/Clear-classification_data4.csv', encoding='latin-1')

# 分离特征和标签
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# 处理数据标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 分离训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

# 使用SMOTE处理不平衡数据
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)


# 定义生成器
def build_generator(input_dim):
    model = Sequential()
    model.add(Dense(128, input_dim=input_dim))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(256))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(512))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(input_dim, activation='tanh'))
    return model


# 定义判别器
def build_discriminator(input_dim):
    model = Sequential()
    model.add(Dense(512, input_dim=input_dim))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(256))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(1, activation='sigmoid'))
    return model


# 构建和编译GAN
def build_gan(generator, discriminator):
    model = Sequential()
    model.add(generator)
    model.add(discriminator)
    return model


# 初始化模型
input_dim = X_resampled.shape[1]
generator = build_generator(input_dim)
discriminator = build_discriminator(input_dim)
discriminator.compile(loss='binary_crossentropy', optimizer=Adam(0.0002, 0.5), metrics=['accuracy'])

# 编译GAN
gan = build_gan(generator, discriminator)
gan.compile(loss='binary_crossentropy', optimizer=Adam(0.0002, 0.5))

# 训练GAN
epochs = 10000
batch_size = 64

for epoch in range(epochs):
    # 生成虚假样本
    noise = np.random.normal(0, 1, (batch_size, input_dim))
    gen_samples = generator.predict(noise)

    # 真实样本
    idx = np.random.randint(0, X_resampled.shape[0], batch_size)
    real_samples = X_resampled[idx]

    # 标签
    real_labels = np.ones((batch_size, 1))
    fake_labels = np.zeros((batch_size, 1))

    # 训练判别器
    d_loss_real = discriminator.train_on_batch(real_samples, real_labels)
    d_loss_fake = discriminator.train_on_batch(gen_samples, fake_labels)
    d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

    # 训练生成器
    g_loss = gan.train_on_batch(noise, real_labels)

    if epoch % 1000 == 0:
        print(f"Epoch {epoch} - D loss: {d_loss[0]} - D acc.: {d_loss[1]} - G loss: {g_loss}")

# 使用生成样本进行训练
noise = np.random.normal(0, 1, (15000, input_dim))
synthetic_samples = generator.predict(noise)

X_train_gan = np.vstack((X_resampled, synthetic_samples))
y_train_gan = np.hstack((y_resampled, np.random.choice([5, 6], size=synthetic_samples.shape[0])))

# 训练分类模型
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train_gan, y_train_gan)

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
plot_correlation_heatmap(pd.DataFrame(X_train_gan, columns=data.columns[:-1]), 'Correlation Matrix - Synthetic Data')


# PCA降维图
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


plot_pca(X_resampled, synthetic_samples)