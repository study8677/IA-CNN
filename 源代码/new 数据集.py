import tensorflow as tf
import numpy as np
import os

# 加载 CIFAR-10 数据集
(x_train, y_train), (_, _) = tf.keras.datasets.cifar10.load_data()

# 初始化新数据集
shiyan2ceshiji_x = []
shiyan2ceshiji_y = []

# 对于每个类别，选择 1000 个样本
num_classes = 10
samples_per_class = 3000

for cls in range(num_classes):
    indices = np.where(y_train == cls)[0]  # 找到当前类别的所有索引
    np.random.shuffle(indices)  # 随机打乱索引
    selected_indices = indices[:samples_per_class]  # 选择前 3000 个索引

    # 将选中的数据添加到新数据集中
    shiyan2ceshiji_x.extend(x_train[selected_indices])
    shiyan2ceshiji_y.extend(y_train[selected_indices].flatten())

# 加载名为 "循环五次.npz" 的文件
npz_data = np.load("循环五次.npz")
x_data = npz_data['X_train_subset']
y_data = npz_data['y_train_subset'].flatten()

for cls in range(num_classes):
    indices = np.where(y_data == cls)[0]  # 找到当前类别的所有索引
    half_indices = indices[::2]  # 使用步长为 2 的切片操作选择每个索引下的数量减半

    # 将选中的数据添加到新数据集中
    shiyan2ceshiji_x.extend(x_data[half_indices.tolist()])
    shiyan2ceshiji_y.extend(y_data[half_indices.tolist()])

    # 将选中的数据添加到新数据集中
    shiyan2ceshiji_x.extend(x_data[half_indices.tolist()])
    shiyan2ceshiji_y.extend(y_data[half_indices.tolist()])
# 将新数据集转换为 NumPy 数组
shiyan2ceshiji_x = np.array(shiyan2ceshiji_x)
shiyan2ceshiji_y = np.array(shiyan2ceshiji_y)

# 创建新数据集存储目录
os.makedirs("shiyan2ceshiji", exist_ok=True)

# 保存新数据集到磁盘（NPZ 格式）
np.savez("shiyan2ceshiji/data.npz", x=shiyan2ceshiji_x, y=shiyan2ceshiji_y)

# 输出每个类别的数量和总数量
class_counts = np.zeros(num_classes, dtype=int)
for cls in range(num_classes):
    class_counts[cls] = np.sum(shiyan2ceshiji_y == cls)
    print(f"类别 {cls}: {class_counts[cls]} 个样本")

total_count = np.sum(class_counts)
print(f"总数量: {total_count} 个样本")