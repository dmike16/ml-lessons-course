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

# load the dataset
dogs_vs_cats = bdc.DogVSCat()
print(dogs_vs_cats)

