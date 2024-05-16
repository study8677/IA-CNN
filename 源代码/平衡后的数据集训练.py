import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten
from keras.utils import to_categorical
from sklearn.metrics import confusion_matrix
from keras.layers import BatchNormalization
from keras.layers import Dropout
import matplotlib.pyplot as plt
import seaborn as sns

# 加载数据集
data_updated = np.load('cifar10_lt_balanced.npz', allow_pickle=True)
x_train_updated = data_updated['x_train_lt']
y_train_updated = data_updated['y_train_lt']

# 将训练集划分为训练集和验证集
x_train, x_val, y_train, y_val = train_test_split(x_train_updated, y_train_updated, test_size=0.2, random_state=42)
# 将标签转换为独热编码
y_train = to_categorical(y_train, 10)
y_val = to_categorical(y_val, 10)

# 计算类别权重
y_train_list = list(np.argmax(y_train, axis=1))
class_weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train_list),
    y=y_train_list)
class_weights_dict = dict(enumerate(class_weights))

# 构建CNN模型
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

# 编译模型
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 训练模型
history = model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=100, batch_size=256, class_weight=class_weights_dict)

# 在验证集上评估模型
y_pred = model.predict(x_val)
y_pred = np.argmax(y_pred, axis=1)
y_val = np.argmax(y_val, axis=1)
accuracy = np.mean(y_pred == y_val)
print('Accuracy: %.3f' % accuracy)

# 计算混淆矩阵
conf_matrix = confusion_matrix(y_val, y_pred)
print('Confusion matrix:\n', conf_matrix)

# 保存模型
model.save("my_model-cifar-lf-2.h5")

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