# -----------------------------------------------------------
# Exampaple of saving and loading tensorflow keras model
#
#
# The model is created usgin tensorflow + tensorflow_dataset  keras + numpy
#
# dmike16, Rome, Italy
# Released under MIT license
# email cipmiky@gmail.com
# -----------------------------------------------------------

import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import os

import bag.dogs_vs_cats as bdc

# load the dataset
dogs_vs_cats = bdc.DogVSCat()
print(dogs_vs_cats)

# Variable to use in building model process
BATCH_SIZE = 32
IMG_SIZE = 224

# prepare the data
train_data_gen = dogs_vs_cats.train_image_generator_flow(BATCH_SIZE, IMG_SIZE)
val_data_gen = dogs_vs_cats.validation_image_generator_flow(BATCH_SIZE, IMG_SIZE)

# Download MobileNew model
URL = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4"
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
EPOCH = 3
history = model.fit(
    train_data_gen,
    epochs=EPOCH,
    validation_data=val_data_gen,
)

class_name = bdc.DogVSCat.labels()
image_batch, label_batch = next(val_data_gen)
pred_batch = model.predict(image_batch)
pred_batch = tf.squeeze(pred_batch)
pred_ids = np.argmax(pred_batch, axis=-1)
pred_class_name = class_name[pred_ids]

print("Label batch ", label_batch)
print("Label predicted ", pred_ids)

curr_dir = os.getcwd()
# Save the model in HDF5 keras model
export_keras_path = "{}/model_dog_cat.h5".format(curr_dir)
model.save(export_keras_path)

# Export as a saved model a tensorflow  object
export_tensorflow_path = "{}/model_dog_cat".format(curr_dir)
tf.saved_model.save(model, export_tensorflow_path)
