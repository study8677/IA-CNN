# 加载图片
from PIL import Image
import numpy as np
from keras.models import load_model

# 加载模型
model = load_model('my_model-2.h5')

image = Image.open('123.jpg')

# 调整大小和格式
image = image.resize((32, 32))
image = np.asarray(image)
image = np.expand_dims(image, axis=0)

# 对图片进行分类
predictions = model.predict(image)
category = np.argmax(predictions)

# 打印分类结果
print('This image belongs to category:', category)

#
# 数据集 训练  2：8    训练集已知标签，0-10大类小类，统计数量，大类小类 选取一个大类  把所有的小类通过图像生成添加像本，使得和最大类样本数量相同，对原来不平衡的图像进行分类。初始的CNN 找出难样本和简单样本找出
# 从大类里选少的
# 计算每一个样本（原来不平衡集里面的所有样本）的交叉熵，根据交叉熵来判断难易，交叉熵的定义，大表示难度比较大。
