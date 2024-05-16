import numpy as np

data = np.load('cifar10_lt.npz')
files = data.files

x_train = data['x_val']
y_train = data['y_val']

# 定义cifar-lf-1的数据集
x_cifar_lf_1 = np.zeros((80, 32, 32, 3), dtype=np.float32)
y_cifar_lf_1 = np.zeros(80, dtype=int)

# 第10类有8个样本
...

# 保存数据集
np.savez('cifar-lf-1.npz', x_train=x_cifar_lf_1, y_train=y_cifar_lf_1)

print(x_cifar_lf_1.shape)
print(y_cifar_lf_1.shape)