import numpy as np
from sklearn.model_selection import train_test_split

# 定义不同类别的采样权重
class_sizes = {
    0: 8000,  # 类别0样本数量为8000
    1: 8000,  # 类别1样本数量为8000
    2: 8000,  # 类别2样本数量为8000
    3: 8000,  # 类别3样本数量为8000
    4: 8000,  # 类别4样本数量为8000
    5: 200,   # 类别5样本数量为200
    6: 100,   # 类别6样本数量为100
    7: 50,    # 类别7样本数量为50
    8: 20,    # 类别8样本数量为20
    9: 10,     # 类别9样本数量为10
}

# 载入 CIFAR-10 数据集
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# 将图像数据归一化到0~1之间
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0
y_train = y_train.reshape(-1) # 将 y_train 从二维数组转换为一维数组

# 将采样权重应用于训练集
class_weights = {k: np.sum(y_train == k) / v for k, v in class_sizes.items()}
sample_weights = np.array([class_weights[y_train[i]] for i in range(len(y_train))], dtype='float32')
sample_weights /= np.max(sample_weights)

# 仅选择部分样本作为每个类别的样本
x_train_lt = []
y_train_lt = []
for class_idx in range(10):
    indices = np.where(y_train == class_idx)[0]
    np.random.shuffle(indices)
    indices = indices[:class_sizes[class_idx]]
    x_train_lt.append(x_train[indices])
    y_train_lt.append(y_train[indices].astype('int32')) # 将 y_train_lt 转换为整数类型的一维数组
x_train_lt = np.concatenate(x_train_lt, axis=0)
y_train_lt = np.concatenate(y_train_lt, axis=0)

# 划分验证集和测试集
x_train_lt, x_val, y_train_lt, y_val = train_test_split(x_train_lt, y_train_lt, test_size=0.2, stratify=y_train_lt, random_state=42, shuffle=True)

# 保存 CIFAR-LT 数据集
np.savez('cifar10_lt.npz', x_train_lt=x_train_lt, y_train_lt=y_train_lt, x_val=x_val, y_val=y_val, x_test=x_test, y_test=y_test)
