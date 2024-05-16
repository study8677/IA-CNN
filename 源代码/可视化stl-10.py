import numpy as np
import matplotlib.pyplot as plt
import torchvision
from torchvision import transforms

# 加载 STL-10 数据集
stl10_train = torchvision.datasets.STL10(
    root='./data', split='train', download=False, transform=transforms.ToTensor()
)

# 可视化函数
def visualize_stl10(dataset, num_images=9):
    figure = plt.figure(figsize=(8, 8))
    cols, rows = 3, 3
    for i in range(1, cols * rows + 1):
        img_idx = np.random.randint(len(dataset))
        img, label = dataset[img_idx]
        figure.add_subplot(rows, cols, i)
        plt.title(f"Label: {label}")
        plt.axis("off")
        plt.imshow(img.permute(1, 2, 0))
    plt.show()

# 显示随机图像
visualize_stl10(stl10_train)