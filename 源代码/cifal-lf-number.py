import numpy as np

# 加载数据集
data = np.load('cifar10_lt.npz')

# 获取训练集标签
y_train_lt = data['y_train_lt']

# 获取标签数量
num_classes = len(np.unique(y_train_lt))

# 获取每个标签的样本数量
label_counts = np.bincount(y_train_lt, minlength=num_classes)

min_label_count = np.min(label_counts)
min_label_index = np.where(label_counts == min_label_count)[0][0]
print(f"最小类的样本类别为：{min_label_index}")
# 打印结果
print(f"标签数量：{num_classes}")
print(f"每个标签的样本数量：{label_counts}")
import matplotlib.pyplot as plt

# 绘制样本数量柱状图
plt.bar(range(num_classes), label_counts)
plt.xticks(range(num_classes), np.unique(y_train_lt))
plt.xlabel('Class')
plt.ylabel('Number of samples')
plt.title('Class distribution of CIFAR-10-LT')
plt.show()
