import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os
from torchvision import datasets, transforms

# 设置字体
font_path = "C:\\Windows\\Fonts\\SimHei.ttf"

if os.path.exists(font_path):
    font = matplotlib.font_manager.FontProperties(fname=font_path)
else:
    font = None

# 加载 STL-10 数据集
stl10_train = datasets.STL10(
    root='./data', split='train', download=True, transform=transforms.ToTensor()
)

# 定义图像增强变换
transform_augmentation = transforms.Compose([
    transforms.RandomRotation(degrees=20),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
])

# 随机选择 5 张图片
num_images = 5
random_indices = np.random.randint(0, len(stl10_train), num_images)
selected_images = [stl10_train[i][0] for i in random_indices]

# 应用图像增强并可视化结果
fig, axs = plt.subplots(2, num_images, figsize=(15, 6))

for i, image in enumerate(selected_images):
    # 将 PyTorch Tensor 转换为 PIL 图像
    image_pil = transforms.ToPILImage()(image)

    # 应用图像增强
    augmented_image = transform_augmentation(image_pil)

    # 将增强后的 PIL 图像转换为 PyTorch Tensor
    augmented_image_tensor = transforms.ToTensor()(augmented_image)

    # 显示原始图像和增强后的图像
    axs[0, i].imshow(image.numpy().transpose((1, 2, 0)))

    if font is not None:
        axs[0, i].set_title(f"原始图像 {i + 1}", fontproperties=font)
    else:
        axs[0, i].set_title(f"Original Image {i + 1}")

    axs[0, i].axis("off")

    axs[1, i].imshow(augmented_image_tensor.numpy().transpose((1, 2, 0)))

    if font is not None:
        axs[1, i].set_title(f"增强后的图像 {i + 1}", fontproperties=font)
    else:
        axs[1, i].set_title(f"Augmented Image {i + 1}")

    axs[1, i].axis("off")

plt.tight_layout()
plt.show()