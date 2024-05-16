import tensorflow as tf
import pickle

# 加载Keras模型
model = tf.keras.models.load_model(' ')

# 序列化为pickle文件
with open('yuxunlian-model.pkl', 'wb') as f:
    pickle.dump(model, f)