import tensorflow as tf
import numpy as np
import os

import numpy as np
import tensorflow as tf

# 加载循环五次.npz数据集
data = np.load('循环五次.npz')

# 从文件中提取数据
X_train_subset = data['X_train_subset']
y_train_subset = data['y_train_subset']

# 将提取的数据分配给x_train和y_train
x_train = X_train_subset
y_train = y_train_subset
# 初始化新数据集
shiyan2ceshiji_x = []
shiyan2ceshiji_y = []

# 对于每个类别，选择 1000 个样本
num_classes = 10
samples_per_class = 3000

for cls in range(num_classes):
    indices = np.where(y_train == cls)[0]  # 找到当前类别的所有索引
    np.random.shuffle(indices)  # 随机打乱索引
    selected_indices = indices[:samples_per_class]  # 选择前 1000 个索引

    # 将选中的数据添加到新数据集中
    shiyan2ceshiji_x.extend(x_train[selected_indices])
    shiyan2ceshiji_y.extend(y_train[selected_indices])

# 将新数据集转换为 NumPy 数组
shiyan2ceshiji_x = np.array(shiyan2ceshiji_x)
shiyan2ceshiji_y = np.array(shiyan2ceshiji_y)

# 创建新数据集存储目录
os.makedirs("shiyan2ceshiji", exist_ok=True)

# 保存新数据集到磁盘（NPZ 格式）
np.savez("shiyan2ceshiji/data.npz", x=shiyan2ceshiji_x, y=shiyan2ceshiji_y)