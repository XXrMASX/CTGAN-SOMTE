import pandas as pd
import numpy as np
from keras.layers import Input, Dense, LeakyReLU
from keras.models import Model
from keras.optimizers import Adam
import xlsxwriter

# 载入数据
data = pd.read_csv('data/LightGBMprocess11.csv')
feature_columns = [col for col in data.columns if col != 'target']

X = data[feature_columns].values
y = data['target'].values

# 定义GAN的参数
latent_dim = 100
gen_units = 50
disc_units = 50

# 生成器模型
def build_generator():
    input = Input(shape=(latent_dim,))
    x = Dense(gen_units, activation=LeakyReLU(0.2))(input)
    output = Dense(len(feature_columns), activation='tanh')(x)
    model = Model(input, output)
    return model

# 判别器模型
def build_discriminator():
    input = Input(shape=(len(feature_columns),))
    x = Dense(disc_units, activation=LeakyReLU(0.2))(input)
    output = Dense(1, activation='sigmoid')(x)
    model = Model(input, output)
    return model

# 生成器
generator = build_generator()
# 判别器
discriminator = build_discriminator()
discriminator.compile(loss='binary_crossentropy', optimizer=Adam(0.0002, 0.5))
# GAN模型
discriminator.trainable = False
z = Input(shape=(latent_dim,))
fake_data = generator(z)
validity = discriminator(fake_data)
combined = Model(z, validity)
combined.compile(loss='binary_crossentropy', optimizer=Adam(0.0002, 0.5))

# 训练GAN
def train(epochs, batch_size=128, save_interval=50):
    for epoch in range(epochs):
        # 训练判别器
        idx = np.random.randint(0, X.shape[0], batch_size)
        real_data = X[idx]
        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        fake_data = generator.predict(noise)

        d_loss_real = discriminator.train_on_batch(real_data, np.ones((batch_size, 1)))
        d_loss_fake = discriminator.train_on_batch(fake_data, np.zeros((batch_size, 1)))
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        # 训练生成器
        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        g_loss = combined.train_on_batch(noise, np.ones((batch_size, 1)))

        if epoch % save_interval == 0:
            print(f"{epoch} [D loss: {d_loss}] [G loss: {g_loss}]")

train(epochs=5000, batch_size=32, save_interval=1000)

# 生成数据
noise = np.random.normal(0, 1, (100, latent_dim))
generated_data = generator.predict(noise)

# 将生成的数据添加到原始数据中
new_data = np.vstack([X, generated_data])
new_indices = list(range(X.shape[0], new_data.shape[0]))
new_df = pd.DataFrame(new_data, columns=feature_columns)

# 保存结果到Excel
writer = pd.ExcelWriter('generated_data.xlsx', engine='xlsxwriter')

def highlight_generated(row):
    return ['background-color: yellow' if row.name in new_indices else '' for _ in row]

highlighted_output = new_df.style.apply(highlight_generated, axis=1)
highlighted_output.to_excel(writer, index=False, sheet_name='Sheet1')

writer.save()
# # 保存结果到Excel
# output = pd.DataFrame(generated_data, columns=feature_columns)
# highlighted_output = output.style.apply(lambda x: ['background: yellow' if x.name in new_indices else '' for i in x])
# with pd.ExcelWriter('generated_data.xlsx', engine='xlsxwriter') as writer:
#     highlighted_output.to_excel(writer, index=False)
#     workbook  = writer.book
#     worksheet = writer.sheets['Sheet1']
#     worksheet.conditional_format('A1:ZZ100', {'type': '2_color_scale'})