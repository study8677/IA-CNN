import numpy as np
import tensorflow as tf

# 加载数据集
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.cifar10.load_data()

# 归一化
X_train, X_test = X_train / 255.0, X_test / 255.0

# one-hot编码
y_train, y_test = tf.keras.utils.to_categorical(y_train, 10), tf.keras.utils.to_categorical(y_test, 10)

# 建立模型
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

# 编译模型
model.compile(loss=tf.keras.losses.categorical_crossentropy,
              optimizer='adam',
              metrics=['accuracy'])


# 定义函数，用于计算交叉熵最大的样本的索引
def get_max_entropy_idx(x, y, model):
    entropies = []
    for i in range(len(x)):
        entropy = -np.sum(y[i] * np.log(model.predict(np.expand_dims(x[i], axis=0))))  # 计算样本i的交叉熵
        entropies.append(entropy)
    max_entropy_idx = np.argmax(entropies)  # 找到交叉熵最大的样本的索引
    return max_entropy_idx

# 定义函数，用于检查数据集是否平衡
def is_balanced(y):
    class_counts = np.sum(y, axis=0)
    min_count = np.min(class_counts)
    max_count = np.max(class_counts)
    if max_count - min_count > 1:  # 如果最多类别的样本数量比最少类别多出超过1个，则数据集不平衡
        return False
    else:
        return True

# 第一次循环，选出与最小类相同数量的样本进行训练
min_samples_per_class = np.min(np.sum(y_train, axis=0))
x_train_balanced = []
y_train_balanced = []
for i in range(10):
    idxs = np.where(y_train[:, i] == 1)[0][:min_samples_per_class]
    x_train_balanced.append(x_train[idxs])
    y_train_balanced.append(y_train[idxs])
x_train_balanced = np.concatenate(x_train_balanced, axis=0)
y_train_balanced = np.concatenate(y_train_balanced, axis=0)
model.fit(x_train_balanced, y_train_balanced, epochs=50, batch_size=64, validation_data=(x_test, y_test))

# 第二次循环，选出与第八类样本相同数量的样本进行训练，并对第九类进行增强，直到数据集平衡为止
while not is_balanced(y_train_balanced):
    for n in range(10):
        # 对于0到7的标签，我们只在每个标签中选出与第八类数量相同的样本
        if n < 8:
            idxs = np.where(y_train[:, n] == 1)[0][:min_samples_per_class]
            x_train_balanced = x_train[idxs]
            y_train_balanced = y_train[idxs]
        # 对于第九类，我们使用增强技术对交叉熵最大的样本进行增强，使其样本数量与第八类相同
        elif n == 8:
            idxs = np.where(y_train[:, n] == 1)[0]
            x_train_n = x_train[idxs]
            y_train_n = y_train[idxs]
            max_entropy_idx = get_max_entropy_idx(x_train_n, y_train_n, model)
            x_augmented = augment_batch([x_train_n[max_entropy_idx]], (32, 32))
            x_train_balanced = np.concatenate([x_train_balanced, x_augmented], axis=0)
            y_train_balanced = np.concatenate([y_train_balanced, y_train_n[max_entropy_idx].reshape((1, 10))], axis=0)
        # 对于第八类和第九类，我们运用图像增强技术使其达到第七类的样本数量
        elif n == 9:
            idxs = np.where(y_train[:, n] == 1)[0]
            x_train_n = x_train[idxs]
            y_train_n = y_train[idxs]
            while np.sum(y_train_balanced[:, 7]) < np.sum(y_train_balanced[:, 9]):
                max_entropy_idx = get_max_entropy_idx(x_train_n, y_train_n, model)
                x_augmented = augment_batch([x_train_n[max_entropy_idx]], (32, 32))
                x_train_n = np.concatenate([x_train_n, x_augmented], axis=0)
                y_train_n = np.concatenate([y_train_n, y_train_n[max_entropy_idx].reshape((1, 10))], axis=0)
            x_train_balanced = np.concatenate([x_train_balanced, x_train_n[:np.sum(y_train_balanced[:, 9])]], axis=0)
            y_train_balanced = np.concatenate([y_train_balanced, y_train_n[:np.sum(y_train_balanced[:, 9])]], axis=0)
        # 对于前面1-6类我们只需要随机挑选出与第七类数量相同的样本
        else:
            idxs = np.where(y_train[:, n] == 1)[0]
            np.random.shuffle(idxs)
            idxs = idxs[:np.sum(y_train_balanced[:, 7])]
            x_train_balanced = np.concatenate([x_train_balanced, x_train[idxs]], axis=0)
            y_train_balanced = np.concatenate([y_train_balanced, y_train[idxs]], axis=0)
    model.fit(x_train_balanced, y_train_balanced, epochs=50, batch_size=64, validation_data=(x_test, y_test))

# 最终得到平衡的数据集
print('Balanced dataset shape:', x_train_balanced.shape, y_train_balanced.shape)