# -----------------------------------------------------------
# Exampaple of  simple linear regression implemented with
# tensorflow Model and wit keras Model
#
#
# dmike16, Rome, Italy
# Released under MIT license
# email cipmiky@gmail.com
# -----------------------------------------------------------

import tensorflow as tf
from collections.abc import Callable
from typing import Any


class SimpleLinearRegression(tf.Module):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.w = tf.Variable(5.0)  # In general variable should be random initialized
        self.b = tf.Variable(0.0)

    def __call__(self, x):
        return x * self.w + self.b

    def train(self, epochs: int, target: tf.Tensor, x: tf.Tensor, optimizer: Callable[
        [Any, tf.Tensor, tf.Tensor, Callable[[tf.Tensor, tf.Tensor], tf.Tensor], float], None],
              loss: Callable[[tf.Tensor, tf.Tensor], tf.Tensor], learning_rate: float):
        # Collect the history of W-values and b-values to plot later
        weights = []
        biases = []
        for epoch in range(epochs):
            print(f"Epoch {epoch:2d}:")
            print("   ", self._report(loss(target, self(x))))
            optimizer(self, target, x, loss, learning_rate)
            # Track this before I update
            weights.append(self.w.numpy())
            biases.append(self.b.numpy())
        return {'w': weights, 'b': biases}

    def _report(self, loss):
        return f"W = {self.w.numpy():1.2f}, b = {self.b.numpy():1.2f}, loss={loss:2.5f}"


class SimpleLinearRegressionKeras(tf.keras.Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.w = tf.Variable(5.0)  # In general variable should be random initialized
        self.b = tf.Variable(0.0)

    def call(self, x):
        return x * self.w + self.b
