import tensorflow as tf
import tensorflow_datasets as tfds
import logging
import numpy as np

np.set_printoptions(precision=3, suppress=True)

# Set tensorflow logger to ERROR level only
tf.get_logger().setLevel(logging.ERROR)
# Disable tensorflow_datasets progressbar
tfds.disable_progress_bar()
# print current tensorflow version
print("tensorflow v{}".format(tf.__version__))

