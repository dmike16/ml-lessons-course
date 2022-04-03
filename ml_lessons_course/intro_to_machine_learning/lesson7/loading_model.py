# -----------------------------------------------------------
# Exampaple of loading tensorflow and keras model
#
#
# The model is created usgin tensorflow + tensorflow_dataset  keras + numpy
#
# dmike16, Rome, Italy
# Released under MIT license
# email cipmiky@gmail.com
# -----------------------------------------------------------

import os
import tensorflow as tf
import tensorflow_hub as hub

curr_dir = os.getcwd()

# Example of loading a saved keras HDFS model
model_path_h5 = "{}/ml_lessons_course/intro_to_machine_learning/lesson7/model_dog_cat.h5".format(curr_dir)
reload_h5 = tf.keras.models.load_model(
    model_path_h5,
    # `custom_objects` tells keras how to load a `hub.KerasLayer`
    custom_objects={'KerasLayer': hub.KerasLayer}
)
reload_h5.summary()

# Load a tensorflow saved model format
model_path_dir = "{}/ml_lessons_course/intro_to_machine_learning/lesson7/model_dog_cat".format(curr_dir)
# reload_sm = tf.saved_model.load(model_path_dir)
# The reload model is different from keras object model. We can predict
# simple colling on a batch file, but we can't use fit, summary or other keras function
# To reload the tensorflow format model, we can use the keras load function with the dir as input
reload_sm_keras = tf.keras.models.load_model(
    model_path_dir,
    # `custom_objects` tells keras how to load a `hub.KerasLayer`
    custom_objects={'KerasLayer': hub.KerasLayer}
)
reload_sm_keras.summary()
