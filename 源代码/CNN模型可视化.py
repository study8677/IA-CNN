from keras.models import Model
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

model = tf.keras.models.load_model('my_model-cifar-lf-2.h5')

def visualize_layer_output(model, layer_index, input_data, n_filters, n_columns=8):
    visualize_model = Model(inputs=model.inputs, outputs=model.layers[layer_index].output)
    layer_output = visualize_model.predict(input_data)

    n_rows = n_filters // n_columns + int(n_filters % n_columns > 0)
    for i in range(n_filters):
        plt.subplot(n_rows, n_columns, i + 1)
        plt.imshow(layer_output[0, :, :, i], cmap='gray')
        plt.axis('off')


# Generate some random input data, or use data from your test set
example_input = np.random.rand(1, 32, 32, 3)

# Visualize the output from each layer
for layer_index, layer in enumerate(model.layers):
    plt.figure(figsize=(16, 4))
    plt.suptitle(f"Layer {layer_index}: {layer.__class__.__name__}", fontsize=14)

    if isinstance(layer, Conv2D) or isinstance(layer, MaxPooling2D):
        n_filters = layer.output_shape[-1]
        visualize_layer_output(model, layer_index, example_input, n_filters)
    elif isinstance(layer, Flatten) or isinstance(layer, Dense):
        # Flatten and Dense layers can't be visualized directly in the same way as Conv2D and MaxPooling2D layers,
        # but you can still explore their output values in a tabular or text format.
        pass
    else:
        print(f"Unknown layer type: {layer.__class__.__name__}")

    plt.show()