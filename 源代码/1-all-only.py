import tensorflow as tf
from tensorflow.python.keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
import cloudpickle
import tensorflow as tf


def random_image_augmentation(image):
    # 随机旋转图片
    image = tf.image.rot90(image, k=tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32))

    # 随机水平翻转图片
    image = tf.image.random_flip_left_right(image)

    # 随机垂直翻转图片
    image = tf.image.random_flip_up_down(image)

    # 随机调整图片亮度
    image = tf.image.random_brightness(image, max_delta=0.1)

    # 随机调整图片对比度
    image = tf.image.random_contrast(image, lower=0.9, upper=1.1)

    # 随机调整图片饱和度
    image = tf.image.random_saturation(image, lower=0.9, upper=1.1)

    # 随机调整图片色相
    image = tf.image.random_hue(image, max_delta=0.1)

    return image
tf.config.run_functions_eagerly(True)

# 定义自定义Adam优化器
class CustomAdamOptimizer(tf.keras.optimizers.Adam):
    def __init__(self, **kwargs):
        super(CustomAdamOptimizer, self).__init__(**kwargs)

# 注册自定义Adam优化器
tf.keras.utils.get_custom_objects()['CustomAdamOptimizer'] = CustomAdamOptimizer

# # 加载数据集
# (X_train, y_train), (X_test, y_test) = tf.keras.datasets.cifar10.load_data()
#
# # 归一化
# X_train, X_test = X_train / 255.0, X_test / 255.0
#
# # one-hot编码
# y_train, y_test = tf.keras.utils.to_categorical(y_train, 10), tf.keras.utils.to_categorical(y_test, 10)
#
# # 建立模型
# model = tf.keras.models.Sequential([
#     tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
#     tf.keras.layers.MaxPooling2D((2, 2)),
#     tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
#     tf.keras.layers.MaxPooling2D((2, 2)),
#     tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
#     tf.keras.layers.MaxPooling2D((2, 2)),
#     tf.keras.layers.Flatten(),
#     tf.keras.layers.Dense(128, activation='relu'),
#     tf.keras.layers.Dense(10, activation='softmax')
# ])
#
# # 编译模型
# model.compile(loss=tf.keras.losses.categorical_crossentropy,
#               optimizer=CustomAdamOptimizer(),
#               metrics=['accuracy'])
#
# # 训练模型
# history = model.fit(X_train, y_train, batch_size=128, epochs=10, validation_data=(X_test, y_test))
#
# # 保存模型
# with open('cifar10_model.pkl', 'wb') as f:
#     cloudpickle.dump(model, f)
#
# # 加载模型
# with open('cifar10_model.pkl', 'rb') as f:
#     pretrained_model = cloudpickle.load(f)
#
# # 加载数据集
# data = np.load('cifar10_lt.npz')
# X_train_lt = data['x_train_lt']
# x_test = data['x_test']
# y_test = data['y_test'].astype(int)
#
# y_train_lt = data['y_train_lt']
#
# # 初始化X_train和y_train
# X_train = X_train_lt
# y_train = tf.keras.utils.to_categorical(y_train_lt, 10)
# # 计算每个类别的样本数量
# class_counts = np.bincount(y_train_lt)
#
# # 输出每个类别的数量
# for label, count in enumerate(class_counts):
#     print(f"Label {label}: {count} samples")
# import matplotlib.pyplot as plt

#
# def visualize_shapes(X_train_lt, y_train_lt, x_test, y_test):
#     shape_labels = ['X_train_lt', 'y_train_lt', 'x_test', 'y_test']
#     shape_values = [X_train_lt.shape[0], y_train_lt.shape[0], x_test.shape[0], y_test.shape[0]]
#
#     fig, ax = plt.subplots()
#     ax.bar(shape_labels, shape_values)
#     ax.set_xlabel('Datasets')
#     ax.set_ylabel('Number of Samples')
#     ax.set_title('Dataset Shape Information')
#
#     for i, v in enumerate(shape_values):
#         ax.text(i - 0.1, v + 100, str(v), fontweight='bold')
#
#     plt.show()

#
# print("X_train_lt shape:", X_train_lt.shape)
# print("y_train_lt shape:", y_train_lt.shape)
# print("x_test shape:", x_test.shape)
# print("y_test shape:", y_test.shape)
#
# visualize_shapes(X_train_lt, y_train_lt, x_test, y_test)
# # 输出数据集的形状信息
# print("X_train_lt shape:", X_train_lt.shape)
# print("y_train_lt shape:", y_train_lt.shape)
# # print("x_test shape:", x_test.shape)
# # print("y_test shape:", y_test.shape)
#
# # 计算每个类别的样本数量
# class_counts = tf.math.bincount(y_train_lt)
#
#
# # 从类别数量减1开始，逐步减小到1，针对每个类别进行处理
# for i in range(5, -1, -1):
#
#     X_train_final = []
#     y_train_final = []
#     # 查找类别i的样本索引
#     category_indices = np.where(y_train_lt == i)[0]
#     # 使用查找到的索引从X_train中提取类别i的样本
#     X_category = X_train[category_indices]
#     # 为类别i创建one-hot编码的标签
#     class_label = tf.keras.utils.to_categorical(np.full(len(category_indices), i), 10)
#     # 使用预训练模型对类别i的样本进行预测
#     y_pred = pretrained_model.predict(X_category)
#     # 计算类别i的样本的交叉熵损失
#     cross_entropy = tf.keras.losses.categorical_crossentropy(class_label, y_pred).numpy()
#
#     X_train_subset_new = []
#     y_train_subset_new = []
#
#     # 遍历0到9的类别
#     # 遍历0到9的类别
#     for j in range(10):
#         # 查找类别j的样本索引
#         category_indices = np.where(y_train_lt == j)[0]
#
#         temp_x_subset = []
#         temp_y_subset = []
#
#         # 更新当前类别j的样本数量，使其等于类别i的样本数量
#         required_samples = class_counts[i]
#
#         # 如果类别j的样本数量大于等于所需的样本数量
#         if class_counts[j] >= class_counts[i]:
#             # 从类别j的样本中随机选择所需数量的样本
#             random_indices = np.random.choice(category_indices, required_samples, replace=False)
#             temp_x_subset.extend(X_train[random_indices])
#             temp_y_subset.extend(tf.keras.utils.to_categorical(np.full(required_samples, j), 10))
#         else:
#             # 如果类别j的样本数量小于当前处理的类别i的样本数量
#             if class_counts[j] < class_counts[i]:
#                 # 对 cross_entropy 进行排序，获取排序后的索引，然后反转索引
#                 descending_sorted_indices = tf.argsort(cross_entropy)[::-1]
#                 valid_indices = descending_sorted_indices[descending_sorted_indices < len(category_indices)]
#
#                 # 获取前两个元素（即最大和次大的交叉熵值的索引）
#                 max_indices = valid_indices[:2]
#
#                 # 使用 max_indices 从 category_indices 中选取元素
#                 max_indices = category_indices[max_indices]
#
#                 # 使用 max_indices 从 X_train 中选取元素
#                 max_samples = X_train[max_indices]
#                 augmented_samples = []
#                 # 通过数据增强生成所需数量的额外样本，以使类别j的样本数量等于类别i的样本数量
#                 for _ in range(required_samples - class_counts[j]):
#                     random_sample = max_samples[np.random.randint(0, 2)]
#                     augmented_sample = random_image_augmentation(random_sample)
#                     augmented_samples.append(augmented_sample)
#
#                 # 将类别j的原始样本和生成的增强样本合并
#                 temp_x_subset.extend(X_train[category_indices])
#                 temp_y_subset.extend(tf.keras.utils.to_categorical(np.full(class_counts[j], j), 10))
#                 temp_x_subset.extend(augmented_samples)
#                 temp_y_subset.extend(tf.keras.utils.to_categorical(np.full(len(augmented_samples), j), 10))
#             else:
#                 # 如果类别j的样本数量已经等于或大于当前处理的类别i的样本数量，不做任何操作
#                 continue
#
#         # 将处理后的类别j的样本添加到新的子集中
#         X_train_subset_new.extend(temp_x_subset)
#         y_train_subset_new.extend(temp_y_subset)
#     # 更新y_train_lt，添加新生成的类别i的样本标签
#     y_train_lt = np.concatenate((y_train_lt, np.full(len(X_train_subset_new), i)), axis=0)
#     # 更新class_counts
#     class_counts = tf.math.bincount(y_train_lt)
#
#     # 将新生成的样本添加到最终的训练集中
#     X_train_final.extend(X_train_subset_new)
#     y_train_final.extend(y_train_subset_new)
#
#     # 更新X_train和y_train，添加新生成的样本
#     X_train = np.concatenate((X_train, np.array(X_train_subset_new)), axis=0)
#     y_train = np.concatenate((y_train, np.array(y_train_subset_new)), axis=0)
#
#
#     # 计算 y_train_final 中每个类别的样本数量
#     y_train_final_int = np.argmax(y_train_final, axis=1)
#     class_counts_final = np.bincount(y_train_final_int)
#
#     # 输出每个类别的样本数量
#     for i, count in enumerate(class_counts_final):
#         print(f"Label {i}: {count} samples")
#
#     # 计算 x_train_final 和 y_train_final 中的总样本数量
#     total_samples_x_train_final = len(X_train_final)
#     total_samples_y_train_final = len(y_train_final)
#
#     # 输出总样本数量
#     print(f"Total samples in x_train_final: {total_samples_x_train_final}")
#     print(f"Total samples in y_train_final: {total_samples_y_train_final}")
#
#     # 使用新生成的样本子集训练预训练模型
# #     pretrained_model.fit(np.array(X_train_subset_new), np.array(y_train_subset_new), batch_size=128, epochs=1)
# #     history = pretrained_model.fit(np.array(X_train_subset_new), np.array(y_train_subset_new), batch_size=128, epochs=1)
# # np.savez('循环五次.npz', X_train_subset=X_train_final, y_train_subset=y_train_final)
# # # 保存预训练模型
# # # pretrained_model.save('fan.h5')
# import numpy as np
# import matplotlib.pyplot as plt
# from keras.models import load_model
# from sklearn.metrics import classification_report, confusion_matrix
# import seaborn as sns
# #
# # # # 加载保存的数据
# # # data = np.load('循环五次.npz')
# # # X_train_subset = data['X_train_subset']
# # # y_train_subset = data['y_train_subset']
# # #
# # # # 加载预训练模型
# model = load_model('fan.h5')
# # # with open('cifar10_model.pkl', 'rb') as f:
# # #     pretrained_model2 = cloudpickle.load(f)
# # #
# # #
# # # # 准备一个函数，用于计算混淆矩阵和分类报告
# def evaluate_model(model, X, y_true):
#     y_pred = model.predict(X)
#     y_pred_labels = np.argmax(y_pred, axis=1)
#     y_true_labels = np.argmax(y_true, axis=1)
#
#     confusion_mat = confusion_matrix(y_true_labels, y_pred_labels)
#     classification_rep = classification_report(y_true_labels, y_pred_labels)
#
#     return confusion_mat, classification_rep
#
#
# # 使用模型1评估数据
# confusion_mat1, classification_rep1 = evaluate_model(pretrained_model1, X_train_subset, y_train_subset)
#
# # 使用模型2评估数据
# confusion_mat2, classification_rep2 = evaluate_model(pretrained_model2, X_train_subset, y_train_subset)
#
# # 绘制两个模型的混淆矩阵热力图
# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
# def print_confusion_matrix(confusion_mat, title):
#     print(title)
#     print(confusion_mat)
#     print()
#
# # 输出混淆矩阵具体数值
# print_confusion_matrix(confusion_mat1, "Confusion Matrix: Model 1")
# print_confusion_matrix(confusion_mat2, "Confusion Matrix: Model 2")
#
# sns.heatmap(confusion_mat1, annot=True, fmt='d', cmap='Blues', ax=ax1)
# ax1.set_title('Confusion Matrix: Model 1')
# ax1.set_xlabel('Predicted Label')
# ax1.set_ylabel('True Label')
#
# sns.heatmap(confusion_mat2, annot=True, fmt='d', cmap='Blues', ax=ax2)
# ax2.set_title('Confusion Matrix: Model 2')
# ax2.set_xlabel('Predicted Label')
# ax2.set_ylabel('True Label')
#
# plt.show()
#
# # 打印两个模型的分类报告
# print("Classification Report: Model 1")
# print(classification_rep1)
#
# # print("Classification Report: Model 2")
# # print(classification_rep2)
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import tensorflow as tf

# # 加载训练数据集
data_train = np.load('cifar10_lt.npz')
X_train_lt = data_train['x_train_lt']
y_train_lt = data_train['y_train_lt']
# 加载 '循环五次.npz' 数据集作为训练数据集
# data_train = np.load('循环五次.npz')
#
# X_train_lt = data_train['X_train_subset']
# y_train_lt = data_train['y_train_subset']

# 加载验证数据集
data_val = np.load('shiyan2ceshiji/data.npz')
X_val = data_val['x']
y_val = data_val['y'].astype(int)

# 初始化X_train和y_train
X_train = X_train_lt
# #不平衡集要打开
y_train = tf.keras.utils.to_categorical(y_train_lt, 10)
# y_train = y_train_lt
# 初始化y_val
y_val = tf.keras.utils.to_categorical(y_val, 10)

# 加载模型
model = load_model('cifar10_model.h5')

# 将模型的优化器替换为自定义优化器
model.optimizer = CustomAdamOptimizer()

# 编译模型
model.compile(optimizer=model.optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# 训练模型
history = model.fit(X_train, y_train, epochs=10, batch_size=128, validation_data=(X_val, y_val))

# 准备一个函数，用于计算混淆矩阵和分类报告
def evaluate_model(model, X, y_true):
    y_pred = model.predict(X)
    y_pred_labels = np.argmax(y_pred, axis=1)
    y_true_labels = np.argmax(y_true, axis=1)

    confusion_mat = confusion_matrix(y_true_labels, y_pred_labels)
    classification_rep = classification_report(y_true_labels, y_pred_labels)

    return confusion_mat, classification_rep

# 使用模型评估数据
confusion_mat, classification_rep = evaluate_model(model, X_val, y_val)

# 输出混淆矩阵
print("Confusion Matrix (Validation Dataset):")
print(confusion_mat)
print()

# 绘制混淆矩阵热力图
fig, ax = plt.subplots(figsize=(10, 10))
sns.heatmap(confusion_mat, annot=True, fmt='d', cmap='Blues', ax=ax)
ax.set_title('Confusion Matrix: Benchmark Model on Balanced Dataset')
ax.set_xlabel('Predicted Label')
ax.set_ylabel('True Label')
plt.show()

# 打印分类报告
print("Classification Report (Validation Dataset): Model")
print(classification_rep)