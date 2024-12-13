import matplotlib.pyplot as plt
import matplotlib.patches as patches

# 创建一个新的图
fig, ax = plt.subplots()

# 绘制模块一
rect1 = patches.Rectangle((0.1, 0.1), 0.2, 0.2, linewidth=1, edgecolor='r', facecolor='none', linestyle='--')
ax.add_patch(rect1)

# 绘制模块二
rect2 = patches.Rectangle((0.4, 0.1), 0.2, 0.2, linewidth=1, edgecolor='g', facecolor='none', linestyle='--')
ax.add_patch(rect2)

# 绘制模块三
rect3 = patches.Rectangle((0.7, 0.1), 0.2, 0.2, linewidth=1, edgecolor='b', facecolor='none', linestyle='--')
ax.add_patch(rect3)

# 设置坐标轴范围
plt.xlim(0, 1)
plt.ylim(0, 0.4)

# 标注模块名称
plt.text(0.2, 0.2, '模块一', ha='center', va='center', fontsize=12)
plt.text(0.5, 0.2, '模块二', ha='center', va='center', fontsize=12)
plt.text(0.8, 0.2, '模块三', ha='center', va='center', fontsize=12)

# 隐藏坐标轴
plt.axis('off')

# 显示图像
plt.show()