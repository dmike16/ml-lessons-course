# -----------------------------------------------------------
# Exampaple of tensors tensorflow usage
#
#
# dmike16, Rome, Italy
# Released under MIT license
# email cipmiky@gmail.com
# -----------------------------------------------------------

import numpy as np
import tensorflow as tf

# Tensors are simple multidim array like numpy.arrays
# all tensors are immutable python objects all of the same type to modify one we need to create a new one

rank0_tensors = tf.constant(4, dtype=tf.float32)
print(rank0_tensors)

rank1_tensors = tf.constant([1, 2, 3], dtype=tf.float32)
print(rank1_tensors)

rank2_tensors = tf.constant([[1, 2, 3], [4, 5, 6]], dtype=tf.float32)
print(rank2_tensors)

rank3_tensors = tf.constant([
    [[1, 3, 2], [2, 3, 4]],
    [[5, 6, 7], [7, 8, 9]]
], dtype=tf.float32)
print(rank3_tensors)

# transform into numpy arrays
print('To numpy array')
print(np.array(rank2_tensors))
print(rank2_tensors.numpy())

# simple math
print('Simple math')
a = tf.constant([[1, 2], [3, 4]])
b = tf.ones([2, 2], dtype=tf.int32)

print(a + b, "\n")  # element-wise addition
print(a * b, "\n")  # element-wise multiplication
print(a @ b, "\n")  # matrix multiplication

c = tf.constant([[4.0, 5.0], [10.0, 1.0]])
print(tf.reduce_max(c))  # Find the largest value
print(tf.argmax(c))  # Find the index of the largest value
print(tf.nn.softmax(c))  # Compute the softmax

# Shape ranke and size
rank4_tensors = tf.zeros([3, 2, 4, 5])
print("Type of every element:", rank4_tensors.dtype)  # one type
print("Number of axes:", rank4_tensors.ndim)  # Number of axis = rank
print("Shape of tensor:", rank4_tensors.shape)  # len of each axis
print("Total number of elements (3*2*4*5): ", tf.size(rank4_tensors).numpy())  # total number of element

# reshaping
x = tf.constant([[1], [2], [3]])
print(x.shape)
x_reshaped = tf.reshape(x, [1, 3])
print(x_reshaped)

# cast
print(tf.cast(x, tf.float16))