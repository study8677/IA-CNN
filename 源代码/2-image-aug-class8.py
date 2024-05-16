import cv2
import numpy as np
import tensorflow as tf

# 加载数据集
data = np.load('cifar10_lt_updated.npz')
x_test = data['x_test']
y_test = data['y_test'].astype(int)
x_9_sorted = data['x_train_lt'][np.where(data['y_train_lt'] == 9)[0]]
y_train_lt = data['y_train_lt']

# 加载保存的模型
model = tf.keras.models.load_model('my_model-cifar-lf-2.h5')

# 在测试集上评估模型
y_pred = model.predict(x_test)
num_classes = 10
y_test_onehot = tf.keras.utils.to_categorical(y_test, num_classes)

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

