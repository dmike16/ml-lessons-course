# -----------------------------------------------------------
# Try to degub Cnn layers
# see https://github.com/anktplwl91/visualizing_convnets
# sse https://towardsdatascience.com/understanding-your-convolution-network-with-visualizations-a4883441533b
#
#
# dmike16, Rome, Italy
# Released under MIT license
# email cipmiky@gmail.com
# -----------------------------------------------------------

"""
This module contains a collections of function used to debug cnn layers
"""

from typing import List

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


class DebugCNN:
    def __init__(self):
        self._image_par_row = 16
        self._input_layer = None
        self._output_layers = []
        self._output_labels = []
        self._output_selections = []

    def config_image_per_row(self, num: int):
        self._image_par_row = num
        return self

    def config_input_layer(self, input: tf.keras.layers.Layer):
        self._input_layer = input
        return self

    def config_output_layers(self, output: List[tf.keras.layers.Layer]):
        self._output_layers = [out.output for out in output]
        return self

    def config_output_labels(self, labels: List[str]):
        self._output_labels = labels
        return self

    def config_output_selections(self, selections: List[int]):
        self._output_selections = selections
        return self

    def visualize_cnn(self, image_test: np.array):
        temp_model = tf.keras.Model(inputs=self._input_layer, outputs=self._output_layers)
        temp_predictions = temp_model.predict(image_test)
        activation_list = [temp_predictions[i] for i in self._output_selections]
        for label, prediction in zip(self._output_labels, activation_list):
            n_features = prediction.shape[-1]
            size = prediction.shape[1]
            n_cols = n_features // self._image_par_row
            display_grid = np.zeros((size * n_cols, self._image_par_row * size))

            for col in range(n_cols):
                for row in range(self._image_par_row):
                    channel_image = prediction[0, :, :, col * self._image_par_row + row]
                    channel_image -= channel_image.mean()
                    channel_image /= channel_image.std()
                    channel_image *= 64
                    channel_image += 128
                    channel_image = np.clip(channel_image, 0, 255).astype('uint8')
                    display_grid[col * size: (col + 1) * size, row * size: (row + 1) * size] = channel_image

            scale = 1. / size
            plt.figure(figsize=(scale * display_grid.shape[1], scale * display_grid.shape[0]))
            plt.title(label)
            plt.grid(False)
            plt.imshow(display_grid, aspect='auto', cmap='plasma')

        plt.tight_layout()
        plt.show()
