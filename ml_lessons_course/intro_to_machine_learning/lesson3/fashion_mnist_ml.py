# -----------------------------------------------------------
# Exampaple of deeplearning model to predict images cloats
# from MNSIT dataser
#
#
# The model is created usgin tensorflow + + tensorflow_dataset  keras + numpy
#
# dmike16, Rome, Italy
# Released under MIT license
# email cipmiky@gmail.com
# -----------------------------------------------------------

import tensorflow as tf
import tensorflow_datasets as tfds

import math
import numpy as np
import matplotlib.pyplot as plt

import bag.fashin_mnist as bfm

# load the dataset
mninst = bfm.FashionMNIST()

# images label name
print("Class names: {}".format(mninst.class_names()))
# explore the dataset
print(mninst)

# process the data [train ,test]
# this step is necessary to normalize the data and is common one
# With tensor flow dataset we can do with pipeline
train_dataset = mninst.ds_train.map(bfm.FashionMNIST.normalize)
test_dataset = mninst.ds_test.map(bfm.FashionMNIST.normalize)
# plot first 25 image from training data
# mninst.plot_images(train_dataset.take(25).cache())

# create the model
#   i. first test with flatten layer -> dense layear with 128 -> dense output layer 10
model = tf.keras.Sequential([
    # Transform a 2d array (28,28) in a 1d array 28*28
    tf.keras.layers.Flatten(input_shape=(28, 28, 1)),
    # Dense hidden layer with 128 neuron + relu as activation function
    tf.keras.layers.Dense(128, activation=tf.nn.relu),
    # tf.keras.layers.Dense(128, activation=tf.nn.relu),
    # Dense output layers with 10 node + sofmax activation function
    tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])
#   2. compile the model with adman optimize
#       + sparse cross entropy function (common in classification problem) + accuracy as metric
model.compile(
    optimizer=tf.keras.optimizers.Adam(0.001),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]
)
#   3. start training the model
BATCH_SIZE = 32
train_dataset = train_dataset.cache().repeat().shuffle(mninst.num_train_examples).batch(BATCH_SIZE)
test_dataset = test_dataset.cache().batch(BATCH_SIZE)

model.fit(train_dataset, epochs=5, steps_per_epoch=math.ceil(mninst.num_train_examples / BATCH_SIZE))
#   4. predict on test images
for test_images, test_labels in test_dataset.take(1):
    np_test_images = tfds.as_numpy(test_images)
    np_test_lables = tfds.as_numpy(test_labels)
    predictions = model.predict(np_test_images)
    mninst.plot_predictions(predictions, np_test_lables, np_test_images)
