import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

confusion_matrix = np.array([
    [735, 24, 49, 28, 34, 7, 9, 4, 78, 32],
    [17, 834, 7, 7, 7, 7, 4, 2, 34, 81],
    [69, 10, 568, 59, 107, 77, 66, 12, 22, 10],
    [16, 12, 82, 476, 102, 191, 57, 20, 22, 22],
    [18, 6, 93, 45, 683, 63, 43, 31, 9, 9],
    [15, 10, 75, 129, 72, 627, 23, 25, 9, 15],
    [8, 5, 57, 65, 60, 32, 752, 6, 8, 7],
    [29, 8, 44, 62, 113, 95, 16, 598, 6, 29],
    [77, 32, 24, 11, 15, 9, 7, 6, 789, 30],
    [34, 90, 17, 7, 15, 14, 5, 4, 33, 781]
])

# 使用 seaborn 绘制热力图
plt.figure(figsize=(10, 8))
sns.heatmap(confusion_matrix, annot=True, fmt="d", cmap="YlGnBu", cbar=False)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix Heatmap")
plt.show()