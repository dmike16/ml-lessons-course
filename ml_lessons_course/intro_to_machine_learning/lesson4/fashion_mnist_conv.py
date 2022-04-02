# -----------------------------------------------------------
# Exampaple of deeplearning model to predict images cloats
# from MNSIT dataset using convolution network
#
#
# The model is created usgin tensorflow + tensorflow_dataset  keras + numpy
#
# dmike16, Rome, Italy
# Released under MIT license
# email cipmiky@gmail.com
# -----------------------------------------------------------
import math

import tensorflow as tf
import tensorflow_datasets as tfds
import bag.fashin_mnist as bfm

# load the dataset
mninst = bfm.FashionMNIST()

# images label name
print("Class names: {}".format(mninst.class_names()))
# explore the dataset
print(mninst)

train_dataset_pipe = mninst.ds_train.map(bfm.FashionMNIST.normalize)
test_dataset_pipe = mninst.ds_test.map(bfm.FashionMNIST.normalize)

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), padding="same", activation=tf.nn.relu, input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D((2, 2), strides=2),
    tf.keras.layers.Conv2D(64, (3, 3), padding="same", activation=tf.nn.relu),
    tf.keras.layers.MaxPooling2D((2, 2), strides=2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation=tf.nn.relu),
    tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(0.001),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]
)

BATCH_SIZE = 32
train_dataset_pipe = train_dataset_pipe.cache().repeat().shuffle(mninst.num_train_examples).batch(BATCH_SIZE)
test_dataset_pipe = test_dataset_pipe.cache().batch(BATCH_SIZE)

model.fit(train_dataset_pipe, epochs=1, steps_per_epoch=math.ceil(mninst.num_train_examples / BATCH_SIZE))

for test_images, test_labels in test_dataset_pipe.take(1):
    np_test_images = tfds.as_numpy(test_images)
    np_test_lables = tfds.as_numpy(test_labels)
    predictions = model.predict(np_test_images)
    mninst.plot_predictions(predictions, np_test_lables, np_test_images)
