import tensorflow as tf
import logging

# Set tensorflow logger to ERROR level only
tf.get_logger().setLevel(logging.ERROR)
# Disable tensorflow_datasets progressbar
import tensorflow_datasets as tfds
tfds.disable_progress_bar()
# print current tensorflow version
print("tensorflow v{}".format(tf.version))

