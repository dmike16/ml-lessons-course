# -----------------------------------------------------------
# Exampaple of deeplearning model to predict color images with CNN model
# from DogsVSCats dataset using convolution network
#
#
# The model is created usgin tensorflow + tensorflow_dataset  keras + numpy
#
# dmike16, Rome, Italy
# Released under MIT license
# email cipmiky@gmail.com
# -----------------------------------------------------------

import bag.dogs_vs_cats as bdc
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# load the dataset
dogs_vs_cats = bdc.DogVSCat()
print(dogs_vs_cats)

# Variable to use in building model process
BATCH_SIZE = 100
IMG_SIZE = 150

# prepare the data
train_data_gen = dogs_vs_cats.train_image_generator_flow(BATCH_SIZE, IMG_SIZE)
val_data_gen = dogs_vs_cats.validation_image_generator_flow(BATCH_SIZE, IMG_SIZE)

# plotting first 5 images from training
# sample_taining_image, _ = next(train_data_gen)
# bdc.DogVSCat.plots_images(sample_taining_image[:5])

# model creation
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation=tf.nn.relu, input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Conv2D(64, (3, 3), activation=tf.nn.relu, ),
    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Conv2D(128, (3, 3), activation=tf.nn.relu, ),
    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation=tf.nn.relu, ),
    tf.keras.layers.Dense(2, activation=tf.nn.softmax, )  # inestead we could use a Dense(1, actiovation=tf.nn.sigmoid)
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(0.001),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    # in case we use a sigmoid function we have to replace with binary_crossentrpy
    metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]
)

model.summary()

# train the model
EPOCH = 15
total_cat_train, total_cat_val = dogs_vs_cats.cats_samples()
total_dog_train, total_dog_val = dogs_vs_cats.dogs_samples()
history = model.fit(
    train_data_gen,
    steps_per_epoch=int(np.ceil((total_cat_train + total_dog_train) / float(BATCH_SIZE))),
    epochs=EPOCH,
    validation_data=val_data_gen,
    validation_steps=int(np.ceil((total_cat_val + total_dog_val) / float(BATCH_SIZE))),
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
