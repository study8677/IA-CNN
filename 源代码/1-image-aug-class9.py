import cv2
import numpy as np
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
import tensorflow as tf
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Conv2D, MaxPool2D , Flatten
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import confusion_matrix
from keras.layers import BatchNormalization
from keras.layers import Dropout
# 加载数据集   预训练  不能用这个数据集来做
data = np.load('cifar10_lt.npz')
x_test = data['x_test']
y_test = data['y_test'].astype(int)
x_9_sorted = data['x_train_lt'][np.where(data['y_train_lt'] == 9)[0]]
y_train_lt = data['y_train_lt']

# 加载保存的模型权重
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    BatchNormalization(momentum=0.9, epsilon=0.00001, center=True, scale=True),
    MaxPool2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    BatchNormalization(momentum=0.9, epsilon=0.00001, center=True, scale=True),
    MaxPool2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    BatchNormalization(momentum=0.9, epsilon=0.00001, center=True, scale=True),
    MaxPool2D((2, 2)),
    Dropout(0.25),
    Flatten(),
    Dense(128, activation='relu'),
    BatchNormalization(momentum=0.9, epsilon=0.00001, center=True, scale=True),
    Dropout(0.25),
    Dense(10, activation='softmax')
])

# 加载权重
model.load_weights('my_model-cifar-lf-1.h5')
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


# 在测试集上评估模型
y_pred = model.predict(x_test)
num_classes = 10
y_test_onehot = tf.keras.utils.to_categorical(y_test, num_classes)
score = model.evaluate(x_test, y_test_onehot, verbose=0)
print(f"Test loss: {score[0]:.4f}")#表示模型在预测测试集时的平均损失。损失值越小，表示模型的预测结果与真实标签之间的差距越小，模型的性能越好
print(f"Test accuracy: {score[1]:.4f}")
# 计算每个类别的交叉熵
cross_entropy = tf.keras.losses.categorical_crossentropy(y_test_onehot, y_pred, label_smoothing=0.1)
cross_entropy = cross_entropy.numpy()

# 输出每个类别及其对应的交叉熵
ce_by_class = []
for i in range(num_classes):
    idx = np.where(y_test == i)[0]
    class_ce = np.mean(cross_entropy[idx])
    ce_by_class.append(class_ce)

# 按照交叉熵从大到小排序
sorted_idx = np.argsort(ce_by_class)[::-1]

# 输出每个类别及其对应的交叉熵
for i in sorted_idx:
    class_ce = ce_by_class[i]
    print(f"类别 {i}，交叉熵: {class_ce:.4f}")

# 获取最后一个类别的索引
class_index = 9
index_last = np.where(y_train_lt == class_index)

# 获取最后一个类别的图像
x_last = data['x_train_lt'][index_last]
y_last = y_train_lt[index_last]
# 选交叉熵最大的前两个
# 应用图像增强来将样本数量增加到16
while len(x_last) < 16:
    for i in range(len(x_last)):
        image = x_last[i]
        # 随机翻转
        flip_code = np.random.randint(-1, 2)
        flipped = cv2.flip(image, flip_code)
        # 旋转
        rows, cols = flipped.shape[:2]
        angle = np.random.randint(-10, 10)
        M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
        rotated = cv2.warpAffine(flipped, M, (cols, rows))
        # 高斯模糊
        ksize = (3, 3)
        sigma = 0.5
        blurred = cv2.GaussianBlur(rotated, ksize, sigma)
        # 调整亮度和对比度
        alpha = 1.2
        beta = 30
        adjusted = cv2.convertScaleAbs(blurred, alpha=alpha, beta=beta)
        # 添加到数据集中
        x_last = np.concatenate((x_last, [adjusted]))
        y_last = np.concatenate((y_last, [class_index]))
        # 如果增强了8个样本，跳出循环
        if len(x_last) >= 16:
            break

# 输出最后一个类别的样本数量
print(f"类别 {class_index}: {len(x_last)} 个样本")

# 获取除最小类别外的其他类别的索引
other_classes = [i for i in range(10) if i != class_index]

# 从每个类别中随机选择16个样本
x_other = []
y_other = []
num_per_class = 16
for c in other_classes:
    idx = np.where(y_train_lt == c)[0]
    selected_idx = np.random.choice(idx, size=num_per_class, replace=False)
    x_other.append(data['x_train_lt'][selected_idx])
    y_other.append(y_train_lt[selected_idx])

# 合并所有样本，生成新的数据集
x_new = np.concatenate((x_last, np.concatenate(x_other)), axis=0)
y_new = np.concatenate((y_last, np.concatenate(y_other)), axis=0)

# 输出新数据集中每个标签的样本数量
for i in range(10):
    count = np.sum(y_new == i)
    print(f"类别 {i}: {count} 个样本")

# 保存新的数据集
data_new = {'x_train_lt': x_new, 'y_train_lt': y_new}
np.savez('cifar10_lt_new-16num.npz', **data_new)