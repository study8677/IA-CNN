import numpy as np
import matplotlib.pyplot as plt

categories = np.arange(1, 11)  # 创建包含 1 到 10 的类别向量
counts = np.full(10, 4000)  # 创建一个向量，每个类别的数量都是 4000

# 创建柱状图
plt.bar(categories, counts)
plt.xlabel('类别')
plt.ylabel('数量')
plt.title('各类别的数量')
plt.show()