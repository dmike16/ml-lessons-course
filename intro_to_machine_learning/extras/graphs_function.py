# -----------------------------------------------------------
# Exampaple of tensorflow graphs
#
#
# dmike16, Rome, Italy
# Released under MIT license
# email cipmiky@gmail.com
# -----------------------------------------------------------

# In Graph execution tensors operation are executed as tensor Graph, that can be saved
# and use in a no python env. The graphs are structures that contain a set of tf.Operation
# which represent units of computation, and tf.Tensors ara data tha flow between units.

import tensorflow as tf
import timeit


# tf.Function can be created or colling the tf.function method or using the tf.function decorator
# They can be called like python function.

@tf.function
def a_regular_function(x, y, b):
    x = tf.matmul(x, y)
    x = x + b
    return x


x1 = tf.constant([[1.0, 2.0]])
y1 = tf.constant([[2.0], [3.0]])
b1 = tf.constant(4.0)

print(a_regular_function(x1, y1, b1).numpy())

# tf.function transform also inner functions

print(tf.autograph.to_code(a_regular_function.python_function))
print(a_regular_function.get_concrete_function(x1, y1, b1).graph.as_graph_def())


# measure preformance

def power(x, y):
    result = tf.eye(10, dtype=tf.int32)
    for _ in range(y):
        result = tf.matmul(x, result)
    return result


x2 = tf.random.uniform(shape=[10, 10], minval=-1, maxval=2, dtype=tf.dtypes.int32)
print("eager exec {}".format(timeit.timeit(lambda: power(x2, 100), number=1000)))

power_as_graph = tf.function(power)
print("graph exec {}".format(timeit.timeit(lambda: power_as_graph(x2, 100), number=1000)))
