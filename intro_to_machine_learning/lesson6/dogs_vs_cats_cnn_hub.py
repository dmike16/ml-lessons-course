# -----------------------------------------------------------
# Exampaple of deeplearning model to predict color images with CNN model
# from DogsVSCats dataset using convolution network + transfer learning
#
#
# The model is created usgin tensorflow + tensorflow_dataset  keras + numpy
#
# dmike16, Rome, Italy
# Released under MIT license
# email cipmiky@gmail.com
# -----------------------------------------------------------
import tensorflow as tf
import tensorflow_hub as hub
import matplotlib.pyplot as plt

import bag.dogs_vs_cats as bdc

URL = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/2"

# load the dataset
dogs_vs_cats = bdc.DogVSCat()
print(dogs_vs_cats)

# Variable to use in building model process
BATCH_SIZE = 32
IMG_SIZE = 224

# prepare the data
train_data_gen = dogs_vs_cats.train_image_generator_flow(BATCH_SIZE, IMG_SIZE)
val_data_gen = dogs_vs_cats.validation_image_generator_flow(BATCH_SIZE, IMG_SIZE)

# Download MobileNet model
feature_extractor = hub.KerasLayer(URL, input_shape=(IMG_SIZE, IMG_SIZE, 3))
feature_extractor.trainable = False  # free the variable of mobileNet model
model = tf.keras.Sequential([
    feature_extractor,
    tf.keras.layers.Dense(2)
])

model.summary()

model.compile(
    optimizer=tf.keras.optimizers.Adam(0.001),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    # in case we use a sigmoid function we have to replace with binary_crossentrpy
    metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]
)

# train the model
EPOCH = 6
history = model.fit(
    train_data_gen,
    epochs=EPOCH,
    validation_data=val_data_gen,
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