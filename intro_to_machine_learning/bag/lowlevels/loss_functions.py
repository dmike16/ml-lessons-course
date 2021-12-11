# -----------------------------------------------------------
# Collections of custom losses functions
#
#
# dmike16, Rome, Italy
# Released under MIT license
# email cipmiky@gmail.com
# -----------------------------------------------------------

"""
This module contains a collections of custom losses function
"""

import tensorflow as tf


def mse(target, prediction):
    return tf.reduce_mean(tf.square(target - prediction))
