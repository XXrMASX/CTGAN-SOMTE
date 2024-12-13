from keras.models import Model
from keras.layers import Input, Conv1D, MaxPooling1D, GlobalAveragePooling1D, Dense, add
from keras.utils import plot_model

# 输入形状
input_shape = (15498, 10)  # (样本数, 时间步, 特征数)

# 类别数量
num_classes = 4

# 输入层
input_tensor = Input(shape=input_shape)

# 1维卷积层
x = Conv1D(filters=64, kernel_size=3, activation='relu', padding='same')(input_tensor)
x = MaxPooling1D(pool_size=2)(x)

# 第一个残差块
y = Conv1D(filters=64, kernel_size=3, activation='relu', padding='same')(x)
y = Conv1D(filters=64, kernel_size=3, activation='relu', padding='same')(y)
y = add([x, y])  # 将残差块的输出于输入相加
x = y

# 第二个残差块
y = Conv1D(filters=64, kernel_size=3, activation='relu', padding='same')(x)
y = Conv1D(filters=64, kernel_size=3, activation='relu', padding='same')(y)
y = add([x, y])  # 将残差块的输出于输入相加
x = y

# 全局平均池化层
x = GlobalAveragePooling1D()(x)

# 全连接层
output_tensor = Dense(num_classes, activation='softmax')(x)

# 定义模型
model = Model(input_tensor, output_tensor)

# 绘制模型图
plot_model(model, to_file='1d_cnn_with_residual_blocks.png', show_shapes=True)