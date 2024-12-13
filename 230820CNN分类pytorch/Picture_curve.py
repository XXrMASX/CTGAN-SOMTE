import math

import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["SimHei"] # 设置字体
plt.rcParams["axes.unicode_minus"] = False # 正常显示负号
fileName = "pic.csv"

data = pd.read_csv(fileName)
font1 = {'family' : 'Times New Roman',
    'size': 18
    }
font2 ={'family' : 'Times New Roman',
    'size': 16}
font3 ={'family' : 'Times New Roman',
    'size': 20}
font4 ={'family' : 'Times New Roman'}
fig = plt.figure(figsize=(12, 10))
trackNum = data.shape[1]
print(trackNum)
a = ['Depth'] + list(data.columns[1:])
col = ['blue','orangered', 'aqua', 'magenta', 'green', 'red', 'purple', 'orange', 'brown', 'pink', 'brown', 'gray', 'black', 'cyan', 'magenta','grey']
for i in range(1, trackNum):
    ax = fig.add_subplot(1, trackNum - 1, i)
    ax.plot(data.iloc[:, i], data.iloc[:, 0], color=col[i - 1],linewidth=0.5)
    ax.set_xlabel(a[i],font=font2)
    plt.grid(True)
    if i == 0:
        #将Depth设置为y轴
        ax.invert_yaxis()
        # 设置y轴范围
        y_min = math.floor(data.iloc[:, 0].min())
        y_max = math.ceil(data.iloc[:, 0].max())
        ax.set_ylim([2400, 3900])
        ax.set_yticks(range(2400, 3900 + 1, 500),font= font1)
        ax.set_ylabel("Depth/m",font=font1)
    else:
        #其他ax部分+
        ax.invert_yaxis()
        ax.set_ylim([2400, 3900])
        plt.yticks([])

    # 设置每个轨道只显示两端值
    x_min = math.floor(data.iloc[:, i].min())
    x_max = math.ceil(data.iloc[:, i].max())
    ax.set_xlim([x_min, x_max])
    ax.set_xticks([x_min, x_max],font=font2)
    ax.tick_params(axis='both', labelsize=14)
    # ax.xaxis.set_label_position('top')

# 在当前图下方添加文字
plt.text(0.55, 0.03, "Logging parameters", ha='center', font=font3, transform=fig.transFigure)
# # 调整子图之间的间距
plt.subplots_adjust(wspace=0.7)

plt.show()