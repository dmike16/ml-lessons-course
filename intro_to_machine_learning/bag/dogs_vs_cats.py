# -----------------------------------------------------------
# Class to manage the DogVSCat dataset
#
#
# The model is created download with keras
#
# dmike16, Rome, Italy
# Released under MIT license
# email cipmiky@gmail.com
# -----------------------------------------------------------

"""
Module containing utility class to prepare plot, train and validate the DogVSCat
data set
"""

import os
from typing import Tuple
from tensorflow import keras

_URL = 'https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip'


class DogVSCat:
    def __init__(self):
        self._dir_path = keras.utils.get_file('cats_and_dogs_filterted.zip', origin=_URL, extract=True)

        base_dir = os.path.join(os.path.dirname(self._dir_path), 'cats_and_dogs_filtered')
        train_dir = os.path.join(base_dir, 'train')
        validation_dir = os.path.join(base_dir, 'validation')

        self._train_cats_dir = os.path.join(train_dir, 'cats')
        self._train_dogs_dir = os.path.join(train_dir, 'cats')
        self._validation_cats_dir = os.path.join(validation_dir, 'cats')
        self._validation_dogs_dir = os.path.join(validation_dir, 'dogs')

    def __repr__(self) -> str:
        return 'DogVSCat()'

    def __str__(self) -> str:
        num_cats_tr, num_cats_vl = self.cats_samples()
        num_dogs_tr, num_dogs_vl = self.dogs_samples()

        return f'DogVSCata datasets with \n' \
               f'   CATS  ---- TRAIN = {num_cats_tr} VAL = {num_cats_vl} \n' \
               f'   DOGS  ---- TRAIN = {num_dogs_tr} VAL = {num_dogs_vl} \n' \
               f'   TOTAL ---- TRAIN = {num_dogs_tr + num_cats_tr} VAL = {num_dogs_vl + num_cats_vl}\n'

    def cats_samples(self) -> Tuple[int, int]:
        num_cats_tr = len(os.listdir(self._train_cats_dir))
        num_cats_vl = len(os.listdir(self._validation_cats_dir))

        return num_cats_tr, num_cats_vl

    def dogs_samples(self) -> Tuple[int, int]:
        num_dogs_tr = len(os.listdir(self._train_dogs_dir))
        num_dogs_vl = len(os.listdir(self._validation_dogs_dir))

        return num_dogs_tr, num_dogs_vl

    @property
    def path(self):
        return self._dir_path

    @property
    def cats_train_path(self):
        return self._train_cats_dir

    @property
    def dogs_train_path(self):
        return self._train_dogs_dir

    @property
    def cats_validation_path(self):
        return self._validation_cats_dir

    @property
    def dogs_validation_path(self):
        return self._validation_dogs_dir
