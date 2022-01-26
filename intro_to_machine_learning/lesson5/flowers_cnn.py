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

# load the data set
data_set = bf.Flower()
print(data_set)

BATCH_SIZE = 100
IMG_SIZE = 150

# prepare the data
train_data_gen = data_set.train_image_generator_flow(BATCH_SIZE, IMG_SIZE)
val_data_gen = data_set.validation_image_generator_flow(BATCH_SIZE, IMG_SIZE)

# plotting first 5 images from training
sample_taining_image, _ = next(train_data_gen)
bf.Flower.plots_images(sample_taining_image[:5])
