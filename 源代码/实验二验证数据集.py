import numpy as np
import matplotlib.pyplot as plt
from tensorflow.python.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import tensorflow as tf

# 自定义 Adam 优化器
class CustomAdamOptimizer(tf.keras.optimizers.Adam):
    def __init__(self, **kwargs):
        super(CustomAdamOptimizer, self).__init__(**kwargs)

# 加载数据集
data = np.load('cifar10_lt.npz')
X_train_lt = data['x_train_lt']
X_test = data['x_test']
y_test = data['y_test'].astype(int)
y_train_lt = data['y_train_lt']

# 初始化X_train和y_train
X_train = X_train_lt
y_train = tf.keras.utils.to_categorical(y_train_lt, 10)

# 加载模型
model = load_model('fan.h5')

# 将模型的优化器替换为自定义优化器
model.optimizer = CustomAdamOptimizer()

# 编译模型
model.compile(optimizer=model.optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# 训练模型
history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# 准备一个函数，用于计算混淆矩阵和分类报告
def evaluate_model(model, X, y_true):
    y_pred = model.predict(X)
    y_pred_labels = np.argmax(y_pred, axis=1)
    y_true_labels = np.argmax(y_true, axis=1)

    confusion_mat = confusion_matrix(y_true_labels, y_pred_labels)
    classification_rep = classification_report(y_true_labels, y_pred_labels)

    return confusion_mat, classification_rep

# 使用模型评估数据
confusion_mat, classification_rep = evaluate_model(model, X_test, y_test)

# 输出混淆矩阵
print("Confusion Matrix (Imbalanced Dataset):")
print(confusion_mat)
print()

# 绘制混淆矩阵热力图
fig, ax = plt.subplots(figsize=(10, 10))
sns.heatmap(confusion_mat, annot=True, fmt='d', cmap='Blues', ax=ax)
ax.set_title('Confusion Matrix: Model on Imbalanced Dataset')
ax.set_xlabel('Predicted Label')
ax.set_ylabel('True Label')
plt.show()

# 打印分类报告
print("Classification Report (Imbalanced Dataset): Model")
print(classification_rep)