import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import font_manager

# 设置默认字体为 SimHei
font_manager.FontProperties(fname='SimHei.ttf')
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
confusion_mat = np.array([[2734, 181, 23, 54, 6, 2, 0, 0, 0, 0],
                          [169, 2802, 2, 25, 1, 1, 0, 0, 0, 0],
                          [1243, 167, 723, 760, 62, 40, 0, 4, 0, 1],
                          [437, 129, 28, 2365, 23, 16, 0, 2, 0, 0],
                          [832, 221, 48, 868, 977, 21, 0, 33, 0, 0],
                          [447, 130, 69, 2028, 71, 238, 0, 17, 0, 0],
                          [728, 576, 47, 1495, 117, 18, 10, 3, 0, 6],
                          [1024, 216, 49, 750, 438, 48, 0, 475, 0, 0],
                          [1907, 847, 13, 216, 17, 0, 0, 0, 0, 0],
                          [781, 1984, 10, 208, 8, 2, 0, 1, 0, 6]])


# 设置绘图样式
sns.set(font_scale=1.2)
plt.figure(figsize=(10, 8))

# 绘制混淆矩阵
ax = sns.heatmap(confusion_mat, annot=True, fmt='d', cmap='Blues', linewidths=.5)

# 设置标题和轴标签
ax.set_title("Benchmark Model", fontsize=20)
ax.set_xlabel('Predicted Labels', fontsize=16)
ax.set_ylabel('True Labels', fontsize=16)

plt.show()