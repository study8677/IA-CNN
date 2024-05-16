import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix
from sklearn.utils import class_weight
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.python.keras.models import Sequential
import matplotlib.pyplot as plt
import seaborn as sns
# 加载数据集
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.cifar10.load_data()

# 将像素值标准化为0到1之间的浮点数
X_train = X_train.astype('float32') / 255
X_test = X_test.astype('float32') / 255

# 将标签转换为独热编码
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

# 将数据集分为训练集和验证集
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
y_train_list = list(np.argmax(y_train, axis=1))
# 计算类别权重
class_weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train_list),
    y=y_train_list)
class_weights_dict = dict(enumerate(class_weights))
# 构建CNN模型
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
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 训练模型
history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=200, batch_size=128, class_weight=class_weights_dict)

# 在测试集上评估模型
y_pred = model.predict(X_test)
y_pred = np.argmax(y_pred, axis=1)
y_test = np.argmax(y_test, axis=1)
accuracy = np.mean(y_pred == y_test)
print('Accuracy: %.3f' % accuracy)

# 计算混淆矩阵
conf_matrix = confusion_matrix(y_test, y_pred)
print('Confusion matrix:\n', conf_matrix)
# 保存模型
model.save("my_model-2.h5")
# 混淆矩阵（Confusion Matrix）是一种常用的分类模型评估方法。它展示了分类模型的预测结果与真实标签之间的对应关系，可以用于分析分类器的性能。

# 混淆矩阵是一个二维数组，其中每行表示真实标签，每列表示预测标签。对角线上的元素表示分类正确的样本数，其他元素表示分类错误的样本数。例如，在这个混淆矩阵中，第一行第一列的789表示真实标签为0且被预测为0的样本数，第二行第一列的36表示真实标签为1但被错误地预测为0的样本数。
# 绘制训练和验证损失图形
plt.figure()
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')

# 绘制混淆矩阵
def plot_confusion_matrix(conf_matrix, labels):
    fig, ax = plt.subplots()
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', ax=ax, cbar=False)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    plt.show()

labels = np.arange(10)  # 更改为您的数据集的类别标签
plot_confusion_matrix(conf_matrix, labels)
plt.legend()
plt.show()