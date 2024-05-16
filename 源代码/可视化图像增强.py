import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

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

# Load image
image_path = './123.jpg'
image = mpimg.imread(image_path)

# Apply random_image_augmentation
augmented_image = random_image_augmentation(image)

# Display original and augmented images
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].imshow(image)
axes[0].set_title('Original Image')
axes[1].imshow(augmented_image)
axes[1].set_title('Augmented Image')
plt.show()