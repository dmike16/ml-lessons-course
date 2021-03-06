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

import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras
from tensorflow.keras import preprocessing

_URL = 'https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip'


class DogVSCat:
    @staticmethod
    def plots_images(images_arr: np.array):
        fig, axes = plt.subplots(1, len(images_arr), figsize=(20, 20))
        axes = axes.flatten()
        for img, ax in zip(images_arr, axes):
            ax.imshow(img)
        plt.tight_layout()
        plt.show()

    @staticmethod
    def labels() -> np.array:
        return np.array(["cat", "dog"])

    def __init__(self):
        self._dir_path = keras.utils.get_file('cats_and_dogs_filterted.zip', origin=_URL, extract=True)

        base_dir = os.path.join(os.path.dirname(self._dir_path), 'cats_and_dogs_filtered')
        train_dir = os.path.join(base_dir, 'train')
        validation_dir = os.path.join(base_dir, 'validation')

        self._train_cats_dir = os.path.join(train_dir, 'cats')
        self._train_dogs_dir = os.path.join(train_dir, 'cats')
        self._validation_cats_dir = os.path.join(validation_dir, 'cats')
        self._validation_dogs_dir = os.path.join(validation_dir, 'dogs')

        self._train_dir = train_dir
        self._validation_dir = validation_dir

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

    def train_image_generator_flow(self, batch_size: int, target_size: int) -> preprocessing.image.DirectoryIterator:
        return DogVSCat._image_genrator_flow(batch_size=batch_size,
                                             target_size=target_size,
                                             directory=self._train_dir,
                                             horizontal_flip=True,
                                             shear_range=0.2,
                                             width_rang=0.2,
                                             zoom_range=0.2,
                                             heigth_range=0.2,
                                             rotation_range=40)

    def validation_image_generator_flow(self, batch_size: int, target_size: int):
        return DogVSCat._image_genrator_flow(batch_size=batch_size, target_size=target_size,
                                             directory=self._validation_dir)

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

    @staticmethod
    def _image_genrator_flow(batch_size: int, target_size: int,
                             directory: str,
                             horizontal_flip: bool = False,
                             rotation_range: int = 0,
                             zoom_range: float = 0.0,
                             shear_range: float = 0.0,
                             width_rang: float = 0.0,
                             heigth_range: float = 0.0) -> preprocessing.image.DirectoryIterator:
        return preprocessing.image \
            .ImageDataGenerator(rescale=1. / 255,
                                horizontal_flip=horizontal_flip,
                                rotation_range=rotation_range,
                                zoom_range=zoom_range,
                                shear_range=shear_range,
                                width_shift_range=width_rang,
                                height_shift_range=heigth_range) \
            .flow_from_directory(directory=directory,
                                 batch_size=batch_size,
                                 shuffle=True,
                                 target_size=(target_size, target_size),
                                 class_mode='binary')
