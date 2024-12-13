import numpy as np
import matplotlib.pyplot as plt

confusion_matrix = np.array([[1497, 5, 0, 22],
                             [12, 44, 0, 14],
                             [2, 3, 13, 6],
                             [48, 1, 0, 333]])

fig, ax = plt.subplots()
cax = ax.matshow(confusion_matrix, cmap='Blues')

plt.colorbar(cax)

for i in range(4):
    for j in range(4):
        if confusion_matrix[i, j] == 1497:
            color = 'white'
            fontsize = 14
        else:
            color = 'black'
            fontsize = 14  # 修改其他单元格的字体大小为12
        text = ax.text(j, i, confusion_matrix[i, j], ha='center', va='center', color=color, fontsize=fontsize)

plt.xlabel('Prediction label', fontsize=14)
plt.ylabel('True label', fontsize=14)
plt.xticks(ticks=[0, 1, 2, 3], labels=["1", "2", "3", "4"], fontsize=14)
plt.yticks(ticks=[0, 1, 2, 3], labels=["1", "2", "3", "4"], fontsize=14)

plt.tight_layout()
plt.show()