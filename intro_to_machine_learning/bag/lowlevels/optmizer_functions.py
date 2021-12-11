# Collections of custom optimizers functions
#
#
# dmike16, Rome, Italy
# Released under MIT license
# email cipmiky@gmail.com
# -----------------------------------------------------------

"""
This module contains a collections of custom optimizers function
"""

import tensorflow as tf
from collections.abc import Callable


def sdg(model, target: tf.Tensor, x: tf.Tensor, loss: Callable[[tf.Tensor, tf.Tensor], tf.Tensor], learning_rate=0.001):
    with tf.GradientTape() as tape:
        current_loss = loss(target, model(x))

    dv = tape.gradient(current_loss, model.variables)
    for i, v in enumerate(model.variables):
        model.variables[i].assign_sub(dv[i] * learning_rate)
