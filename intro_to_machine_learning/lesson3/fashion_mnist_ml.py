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

# # images label name
print("Class names: {}".format(mninst.class_names()))

# # explore the dataset
print(mninst)
