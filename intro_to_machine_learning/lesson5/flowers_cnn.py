# -----------------------------------------------------------
# Exampaple of deeplearning model to predict color images with CNN model
# from Flower dataset using convolution network
#
#
# The model is created usgin tensorflow + tensorflow_dataset  keras + numpy
#
# dmike16, Rome, Italy
# Released under MIT license
# email cipmiky@gmail.com
# -----------------------------------------------------------

import bag.flowers as bf
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from tensorflow import keras
from tensorflow.keras import layers

# load the data set
data_set = bf.Flower()
print(data_set)

BATCH_SIZE = 100
IMG_SIZE = 150

# prepare the data
train_data_gen = data_set.train_image_generator_flow(BATCH_SIZE, IMG_SIZE)
val_data_gen = data_set.validation_image_generator_flow(BATCH_SIZE, IMG_SIZE)

# plotting first 5 images from training
# sample_taining_image, _ = next(train_data_gen)
# bf.Flower.plots_images(sample_taining_image[:5])

# Create CNN model with functional API
inputs = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
# first Conv layer
x = layers.Conv2D(8, (3, 3), padding="same", activation=tf.nn.relu, name="first-conv")(inputs)
x = layers.MaxPooling2D(pool_size=(2, 2), strides=2)(x)
# second Conv layer
x = layers.Conv2D(16, (3, 3), padding="same", activation=tf.nn.relu, name="second-conv")(x)
x = layers.MaxPooling2D(pool_size=(2, 2), strides=2)(x)
# thrid Conv layer
x = layers.Conv2D(32, (3, 3), padding="same", activation=tf.nn.relu, name="third-conv")(x)
x = layers.MaxPooling2D(pool_size=(2, 2), strides=2)(x)
# flatten layer
x = layers.Dropout(0.2)(x)
x = layers.Flatten()(x)
# dropout layer
x = layers.Dense(512, activation=tf.nn.relu)(x)
# Output layer
x = layers.Dropout(0.2)(x)
outputs = layers.Dense(5, activation=keras.activations.softmax)(x)

model = keras.Model(inputs=inputs, outputs=outputs, name="flower_photos")
model.compile(
    optimizer=keras.optimizers.Adam(0.001),
    loss=keras.losses.SparseCategoricalCrossentropy(),
    metrics=[keras.metrics.SparseCategoricalAccuracy()]
)
model.summary()

# train the model
EPOCH = 10
history = model.fit(
    train_data_gen,
    steps_per_epoch=int(np.ceil(train_data_gen.n / float(BATCH_SIZE))),
    epochs=EPOCH,
    validation_data=val_data_gen,
    validation_steps=int(np.ceil(val_data_gen.n / float(BATCH_SIZE))),
)

# view results
acc = history.history['sparse_categorical_accuracy']
val_acc = history.history['val_sparse_categorical_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(EPOCH)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()
