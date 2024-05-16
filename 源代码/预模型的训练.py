import tensorflow as tf
from tensorflow.python.keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
import cloudpickle
import seaborn as sns
from sklearn.metrics import confusion_matrix

class CustomAdamOptimizer(tf.keras.optimizers.Adam):
    def __init__(self, **kwargs):
        super(CustomAdamOptimizer, self).__init__(**kwargs)

# 注册自定义Adam优化器
tf.keras.utils.get_custom_objects()['CustomAdamOptimizer'] = CustomAdamOptimizer

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
              optimizer=CustomAdamOptimizer(),
              metrics=['accuracy'])

# 训练模型
history = model.fit(X_train, y_train, batch_size=128, epochs=50, validation_data=(X_test, y_test))

# 绘制损失 (Loss) 和准确率 (Accuracy) 随着 epoch 变化的折线图
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.show()

# 计算混淆矩阵
y_pred = model.predict(X_test)
y_pred_labels = np.argmax(y_pred, axis=1)
y_test_labels = np.argmax(y_test, axis=1)
confusion_mat = confusion_matrix(y_test_labels, y_pred_labels)

# 绘制混淆矩阵的热力图
plt.figure(figsize=(10, 10))
sns.heatmap(confusion_mat, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

# 保存模型
with open('cifar10_model.pkl', 'wb') as f:
    cloudpickle.dump(model, f)