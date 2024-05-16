import numpy as np
import matplotlib.pyplot as plt

# 定义混淆矩阵
confusion_matrix_100 = np.array([[735, 24, 49, 28, 34, 7, 9, 4, 78, 32],
                                 [17, 834, 7, 7, 7, 7, 4, 2, 34, 81],
                                 [69, 10, 568, 59, 107, 77, 66, 12, 22, 10],
                                 [16, 12, 82, 476, 102, 191, 57, 20, 22, 22],
                                 [18, 6, 93, 45, 683, 63, 43, 31, 9, 9],
                                 [15, 10, 75, 129, 72, 627, 23, 25, 9, 15],
                                 [8, 5, 57, 65, 60, 32, 752, 6, 8, 7],
                                 [29, 8, 44, 62, 113, 95, 16, 598, 6, 29],
                                 [77, 32, 24, 11, 15, 9, 7, 6, 789, 30],
                                 [34, 90, 17, 7, 15, 14, 5, 4, 33, 781]])

confusion_matrix_200 = np.array([[710, 13, 67, 18, 31, 9, 10, 7, 88, 47],
                                 [20, 767, 4, 11, 5, 9, 14, 9, 32, 129],
                                 [68, 3, 570, 91, 84, 69, 52, 31, 16, 16],
                                 [23, 11, 67, 525, 84, 147, 63, 37, 23, 20],
                                 [11, 3, 79, 76, 689, 35, 37, 51, 12, 7],
                                 [13, 3, 79, 148, 65, 577, 27, 67, 7, 14],
                                 [8, 8, 52, 86, 60, 30, 726, 9, 7, 14],
                                 [18, 1, 44, 56, 74, 75, 7, 706, 3, 16],
                                 [65, 25, 20, 18, 6, 8, 8, 7, 803, 40],
                                 [37, 60, 18, 19, 8, 5, 6, 9, 36, 802]])


# 计算查准率
def calculate_precision(confusion_matrix):
    num_classes = confusion_matrix.shape[0]
    precision = np.zeros(num_classes)

    for i in range(num_classes):
        TP = confusion_matrix[i, i]
        FP = np.sum(confusion_matrix[:, i]) - TP
        precision[i] = TP / (TP + FP)

    return precision


precision_100 = calculate_precision(confusion_matrix_100)
precision_200 = calculate_precision(confusion_matrix_200)

# 设置柱状图的位置和宽度
N = 10
ind = np.arange(N)
width = 0.35

# 创建柱状图
fig, ax = plt.subplots()
rects1 = ax.bar(ind, precision_100, width, label='迭代100次')
rects2 = ax.bar(ind + width, precision_200, width, label='迭代200次')

# 添加标签、标题和图例
ax.set_ylabel('查准率')
ax.set_title('100次迭代与200次迭代查准率比较')
ax.set_xticks(ind + width / 2)
ax.set_xticklabels(('类别0', '类别1', '类别2', '类别3', '类别4', '类别5', '类别6', '类别7', '类别8', '类别9'))
ax.legend()


# 添加数值标签
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{:.2f}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom')


autolabel(rects1)
autolabel(rects2)

# 显示图形
plt.show()
# 输出查准率
print("迭代100次的查准率：", precision_100)
print("迭代200次的查准率：", precision_200)