# -----------------------------------------------------------
# Exampaple of deeplearning model to compute the tranformation
# from celcius to fharenhit degrees.
#
# The model is created usgin tensorflow + keras + numpy
#
# dmike16, Rome, Italy
# Released under MIT license
# email cipmiky@gmail.com
# -----------------------------------------------------------

import tensorflow as tf
from tensorflow import keras
import numpy as np
import logging
import matplotlib.pyplot as plt

tf.get_logger().setLevel(logging.ERROR)

def plot_history(hst):
    """ plot history utilty function
     args:
        keras history

    """
    plt.xlabel("Epoch number")
    plt.ylabel("Loss magnitude")
    plt.plot(hst.history['loss'])
    plt.show()


# Setup traingin data
celsius_q = np.array([-40, -10,  0,  8, 15, 22,  38],  dtype=float)
fahrenheit_a = np.array([-40,  14, 32, 46, 59, 72, 100], dtype=float)

for i, e in enumerate(celsius_q):
    print("{} C = {} F".format(e, fahrenheit_a[i]))

# Create the keras model
# The model is a simple dense layer with one neuron

# shape of our fetures in this case a one dimensional array
input_shape = [1]
# number of variable that our model need to learn
units = 1
# Create keras dense layer
l0 = keras.layers.Dense(units=units, input_shape=input_shape)
# create a sequential model taking in input a list of layers
model = keras.Sequential([l0])
# compile the model passing a loss function + an optimizer function with a
# learing rate
model.compile(loss='mean_squared_error', optimizer=keras.optimizers.Adam(0.1))
print("Finished traingin the model")
# train the model psssing an iteration count epoch
history = model.fit(celsius_q, fahrenheit_a, epochs=600, verbose=False)
# plot_history(history)
predicted_F = model.predict([100.0])
print("100C ~= {}".format(predicted_F))

# print the internal layers variables
print("Variables used by the layer {}".format(l0.get_weights()))


