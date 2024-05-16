import numpy as np
import cloudpickle
from keras.models import load_model
from keras.utils import np_utils

# 加载预训练模型
with open('cifar10_model.pkl', 'rb') as f:
    pretrained_model = cloudpickle.load(f)

# 加载数据集
data = np.load('循环五次.npz')
X_test = data['X_test']
y_test = data['y_test']

# 对标签进行 one-hot 编码
num_classes = len(np.unique(y_test))
y_test = np_utils.to_categorical(y_test, num_classes)

# 预处理数据（如有必要）
# 这里的预处理应与训练模型时相同
# 例如：X_test = X_test.astype('float32') / 255.0

# 在测试数据集上评估模型性能
score = pretrained_model.evaluate(X_test, y_test, verbose=1)
print('Test loss:', score[0])
print('Test accuracy:', score[1])