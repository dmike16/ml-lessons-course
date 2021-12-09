# -----------------------------------------------------------
# Class to manage the MNIST dataset
#
#
# The model is created usgin tensorflow + + tensorflow_dataset  keras + numpy
#
# dmike16, Rome, Italy
# Released under MIT license
# email cipmiky@gmail.com
# -----------------------------------------------------------

"""
Module containing utility class to prepare plot, train and validate the MNIST
fashion ML problem
"""

from typing import List, Tuple

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt


class FashionMNIST:
    @staticmethod
    def normalize(images, label) -> Tuple[tf.Tensor, tf.Tensor]:
        images = tf.cast(images, tf.float32)
        images /= 255
        return images, label

    @property
    def ds_train(self) -> tf.data.Dataset:
        return self._train

    @property
    def ds_test(self) -> tf.data.Dataset:
        return self._test

    @property
    def info(self):
        return self._info

    @property
    def num_train_examples(self) -> int:
        return self.info.splits['train'].num_examples

    @property
    def num_test_examples(self) -> int:
        return self.info.splits['test'].num_examples

    def plot_image(self, image: tf.data.Dataset, label: int, show_plot=False):
        np_image = tfds.as_numpy(image).reshape((28, 28))
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(np_image, cmap=plt.cm.binary)
        plt.xlabel(self.class_names()[label])
        if show_plot:
            plt.show()

    def plot_images(self, images: tf.data.Dataset):
        plt.figure(figsize=(10, 10))
        for i, (image, label) in enumerate(images):
            plt.subplot(5, 5, i + 1)
            self.plot_image(image, label)
        plt.show()

    def plot_predicted_image(self, predictions: np.array, true_label: int, image: np.array):
        plt.grid(False)
        plt.xticks([])
        plt.yticks([])
        img = image[..., 0]
        plt.imshow(img, cmap=plt.cm.binary)
        predicted_label = np.argmax(predictions)
        color = 'blue' if predicted_label == true_label else 'red'
        plt.xlabel("{} {:2.0f}% ({})".format(
            self.class_names()[predicted_label],
            100 * np.max(predictions),
            self.class_names()[true_label],
            color=color
        ))

    def plot_predicted_values(self, predictions: np.array, label: int):
        plt.grid(False)
        plt.xticks([])
        plt.yticks([])
        thisplot = plt.bar(range(10), predictions, color='#777777')
        plt.ylim([0, 1])
        pred_label = np.argmax(predictions)

        thisplot[pred_label].set_color('red')
        thisplot[label].set_color('blue')

    def plot_predictions(self, predictions: np.array, labels: np.array, test_image: np.array):
        num_rows = 5
        num_cols = 3
        num_images = num_cols * num_rows
        plt.figure(figsize=(2 * 2 * num_cols, 2 * num_rows))
        for i in range(num_images):
            plt.subplot(num_rows, 2 * num_cols, 2 * i + 1)
            self.plot_predicted_image(predictions[i], labels[i], test_image[i])
            plt.subplot(num_rows, 2 * num_cols, 2 * i + 2)
            self.plot_predicted_values(predictions[i], labels[i])
        plt.show()

    def class_names(self) -> List:
        return self.info.features['label'].names

    def __init__(self) -> None:
        ds, metainfo = tfds.load('fashion_mnist', as_supervised=True, with_info=True)
        self._info = metainfo
        self._train = ds['train']
        self._test = ds['test']

    def __repr__(self) -> str:
        return f'FashionMNIST()'

    def __str__(self) -> str:
        return f'FashionMNIST data set with {self.num_train_examples} train and {self.num_test_examples} test'
