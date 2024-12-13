import numpy as np
import matplotlib.pyplot as plt

confusion_matrix = np.array([[1896, 12, 0, 3, 9, 0],
                             [1, 238, 0, 2, 1, 0],
                             [6, 0, 309, 6, 0, 0],
                             [3, 6, 0, 553, 0, 0],
                             [0, 6, 0, 9, 252, 0],
                             [5, 0, 0, 0, 0, 402]])

fig, ax = plt.subplots()
cax = ax.matshow(confusion_matrix, cmap='Blues')

plt.colorbar(cax)

for i in range(6):
    for j in range(6):
        if confusion_matrix[i, j] == 1896:
            color = 'white'
            fontsize = 14
        else:
            color = 'black'
            fontsize = 14  # 修改其他单元格的字体大小为12
        text = ax.text(j, i, confusion_matrix[i, j], ha='center', va='center', color=color, fontsize=fontsize)

plt.xlabel('Prediction label', fontsize=14)
plt.ylabel('True label', fontsize=14)
plt.xticks(ticks=[0, 1, 2, 3, 4, 5], labels=["1", "2", "3", "4", "5", "6"], fontsize=14)
plt.yticks(ticks=[0, 1, 2, 3, 4, 5], labels=["1", "2", "3", "4", "5", "6"], fontsize=14)

plt.tight_layout()
plt.show()
