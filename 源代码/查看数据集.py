import numpy as np
data = np.load('循环五次.npz')
print(data.files)
x_new_test = data['X_train_subset']
y_new_test = data['y_train_subset']
print("y_new_test 形状:", y_new_test.shape)

