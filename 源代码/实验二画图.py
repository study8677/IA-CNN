import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
confusion_mat = np.array([
    [2779, 187, 6, 27, 0, 1, 0, 0, 0, 0],
    [133, 2853, 2, 10, 1, 1, 0, 0, 0, 0],
    [1516, 287, 623, 532, 15, 27, 0, 0, 0, 0],
    [628, 342, 38, 1968, 7, 13, 1, 3, 0, 0],
    [1211, 542, 110, 704, 393, 10, 0, 30, 0, 0],
    [575, 342, 96, 1746, 25, 206, 0, 10, 0, 0],
    [910, 1107, 67, 869, 12, 3, 31, 0, 0, 1],
    [1274, 547, 78, 496, 164, 39, 1, 400, 0, 1],
    [2048, 809, 8, 132, 3, 0, 0, 0, 0, 0],
    [727, 2140, 3, 122, 4, 0, 0, 0, 0, 4]
])

# 绘制混淆矩阵热力图
fig, ax = plt.subplots(figsize=(10, 10))
sns.heatmap(confusion_mat, annot=True, fmt='d', cmap='Blues', ax=ax)
ax.set_title('Balanced-Cifar-10-LT Dataset')
ax.set_xlabel('Predicted Label')
ax.set_ylabel('True Label')
plt.show()