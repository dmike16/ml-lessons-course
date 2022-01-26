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
