# -----------------------------------------------------------
# Exampaple of a line dataset with gaussian noise
# used like an input for our custom model
#
#
# dmike16, Rome, Italy
# Released under MIT license
# email cipmiky@gmail.com
# -----------------------------------------------------------

"""
This module contains a class that represent a line with some gaussian noises
"""

import tensorflow as tf
import matplotlib.pyplot as plt


class LineGussianNoise:
    def __init__(self, w: float, b: float, num_example=200):
        self._w = w
        self._b = b
        x = tf.linspace(-2, 2, num_example)
        x = tf.cast(x, tf.float32)
        noise = tf.random.normal(shape=[num_example])
        self._x = x
        self._y = self.f(x) + noise

    @property
    def x(self):
        return self._x

    @property
    def y(self) -> tf.Tensor:
        return self._y

    @property
    def w(self) -> float:
        return self._w

    @property
    def b(self) -> float:
        return self._b

    def f(self, x):
        return x * self._w + self._b

    def plot(self):
        plt.plot(self._x, self._y, '.')
        plt.show()

    def plot_with_predictions(self, prediction: tf.Tensor):
        plt.plot(self._x, self._y, '.', label='Data')
        plt.plot(self._x, self.f(self._x), label="Ground truth")
        plt.plot(self._x, prediction, label='Predictions')
        plt.legend()
        plt.show()

