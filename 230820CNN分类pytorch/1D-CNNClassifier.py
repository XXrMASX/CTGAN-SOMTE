import torch
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
import numpy as np
from util import *

# 定义残差结构
class ResidualBlock(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size=3):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv1d(input_channels, output_channels, kernel_size, padding=1)
        self.conv2 = nn.Conv1d(output_channels, output_channels, kernel_size, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out += residual
        out = self.relu(out)
        return out

# 定义多尺度滑动池化方法
class MultiScaleMaxPool1d(nn.Module):
    def __init__(self, kernel_sizes):
        super(MultiScaleMaxPool1d, self).__init__()
        self.kernel_sizes = kernel_sizes

    def forward(self, x):
        pool_outs = []
        for kernel_size in self.kernel_sizes:
            out = F.max_pool1d(x, kernel_size=kernel_size)
            pool_outs.append(out)
        return torch.cat(pool_outs, dim=1)

# 定义 CNNClassifier 模型
class CNNClassifier(nn.Module):
    def __init__(self, input_channels, num_classes):
        super(CNNClassifier, self).__init__()
        # 输出通道，设置为相对较大的值，例如50、100或更多
        self.conv1 = nn.Conv1d(channels, 50, kernel_size=3)  # 100是输出通道数，kernel_size不能大于数据的维度 本来是3
        self.residual_block = ResidualBlock(50, 50)  # 添加残差块
        self.fc = nn.Linear(50, num_classes)

    def forward(self, x, multiscalenet = True):
        x = torch.relu(self.conv1(x))
        if multiscalenet:
            x = self.multi_scale_pool(x)
        else:
            x = F.max_pool1d(x, kernel_size=3, stride=2)  # 使用普通的 max pooling
        x = self.residual_block(x)  # 通过残差块
        x = x.mean(dim=2)  # 维度缩减
        x = self.fc(x)
        return x

# #定义训练函数
# def train_model():
#     # 训练模型
#     train_losses = []
#     train_accs = []
#
#
#     for epoch in range(num_epochs):
#         model.train()
#         optimizer.zero_grad()    #梯度清零
#         outputs = model(X_train.unsqueeze(1))
#         #print(X_train.unsqueeze(1))
#         #outputs = model(X_train) # 前向传播
#         loss = criterion(outputs, torch.argmax(y_train, dim=1))
#         loss.backward() # 反向传播和优化
#         optimizer.step()
#
#         # 计算训练准确率
#         _, predicted = torch.max(outputs, 1)
#         train_acc = accuracy_score(torch.argmax(y_train, dim=1).numpy(), predicted.numpy())
#
#         # 记录损失和准确率
#         train_losses.append(loss.item())
#         train_accs.append(train_acc)
#
#         if (epoch + 1) % 10 == 0:
#             print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item():.4f}, Train Acc: {train_acc:.4f}')
#
#     return train_losses,train_accs

# 定义训练函数
def train_model():
    # 训练模型
    train_losses = []
    train_accs = []
    val_losses = []  # 添加一个验证集的loss列表
    val_accs = []  # 添加一个验证集的准确率列表

    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()  # 梯度清零
        outputs = model(X_train.unsqueeze(1))
        loss = criterion(outputs, torch.argmax(y_train, dim=1))
        loss.backward()  # 反向传播和优化
        optimizer.step()

        # 计算训练集准确率
        _, predicted = torch.max(outputs, 1)
        train_acc = accuracy_score(torch.argmax(y_train, dim=1).numpy(), predicted.numpy())
        train_losses.append(loss.item())
        train_accs.append(train_acc)

        # 在验证集上进行评估
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val.unsqueeze(1))
            val_loss = criterion(val_outputs, torch.argmax(y_val, dim=1))
            val_losses.append(val_loss.item())
            _, val_predicted = torch.max(val_outputs, 1)
            val_acc = accuracy_score(torch.argmax(y_val, dim=1).numpy(), val_predicted.numpy())
            val_accs.append(val_acc)

        if (epoch + 1) % 10 == 0:
            print(
                f'Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item():.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss.item():.4f}, Val Acc: {val_acc:.4f}')

    return train_losses, train_accs, val_losses, val_accs


#定义测试函数
def test_model():
    # 在测试集上评估模型
    model.eval()
    with torch.no_grad():
        # outputs = model(X_test)
        outputs = model(X_test.unsqueeze(1))
        _, predicted = torch.max(outputs, 1)
        #print(predicted)
        test_acc = accuracy_score(torch.argmax(y_test, dim=1).numpy(), predicted.numpy())
        print("test_acc",test_acc)
    return  test_acc,predicted

if __name__=='__main__':
    # X_train,y_train,X_test,y_test = data_load()    #加载数据
    X_train, y_train, X_val, y_val, X_test, y_test = data_load()  # 加载数据
    print(X_train[0])
    # 设置参数
    channels = 1  #
    num_classes = 8  # 分类类别数
    num_epochs = 500

    # 创建模型实例
    model = CNNClassifier(channels, num_classes)

    optimizer = optim.Adam(model.parameters(), lr=0.001) # 优化器
    criterion = nn.CrossEntropyLoss()               # 损失函数
    # train_losses,train_accs = train_model()        # 训练
    # test_accs, predicted = test_model()            # 测试
    train_losses, train_accs, val_losses, val_accs = train_model()  # 训练
    test_acc, predicted = test_model()  # 测试

    pltloss(train_losses)                           # 绘训练集损失
    labels = np.argmax(y_test, axis=1)                 # 计算实际值
    drawScatter([labels, predicted], ['true', 'pred'])   # 绘制散点图
    plot_confusion(labels,predicted)                      # 绘制混淆矩阵

    result_data = {
        'train_losses': train_losses,
        'train_accs': train_accs,
        'val_losses': val_losses,
        'val_accs': val_accs
    }
    result_df = pd.DataFrame(result_data)
    result_df.to_excel('training_results2.xlsx', index=False)
    # 将train_losses写入excel
    train_losses_df = pd.DataFrame(train_losses, columns=['train_losses'])
    train_losses_df.to_excel('train_losses2.xlsx', index=False)

    # 将labels和predicted写入excel
    data = {'labels': labels, 'predicted': predicted}
    labels_predicted_df = pd.DataFrame(data)
    labels_predicted_df.to_excel('labels_predicted2.xlsx', index=False)

