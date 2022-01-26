# -----------------------------------------------------------
# Class to manage the Flower dataset
#
#
# The model is created download with keras
#
# dmike16, Rome, Italy
# Released under MIT license
# email cipmiky@gmail.com
# -----------------------------------------------------------

"""
Module containing utility class to prepare plot, train and validate the Flower
data set
"""
import os
import glob
import shutil
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import preprocessing
from typing import List, Tuple

_URL = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"


class Flower:
    def __init__(self):
        self._dir_path = keras.utils.get_file('flower_photos.zip', origin=_URL, extract=True)
        self._classes = ['roses', 'daisy', 'dandelion', 'sunflowers', 'tulips']

        base_dir = os.path.join(os.path.dirname(self._dir_path), 'flower_photos')
        train_dir = os.path.join(base_dir, 'train')
        validation_dir = os.path.join(base_dir, 'val')

        _split_train_validation_set(base_dir, self._classes)

        # train dir
        self._train_roses_dir = os.path.join(train_dir, 'roses')
        self._train_daisy_dir = os.path.join(train_dir, 'daisy')
        self._train_dandelion_dir = os.path.join(train_dir, 'dandelion')
        self._train_sunflowers_dir = os.path.join(train_dir, 'sunflowers')
        self._train_tulips_dir = os.path.join(train_dir, 'tulips')
        # validation dir
        self._val_roses_dir = os.path.join(validation_dir, 'roses')
        self._val_daisy_dir = os.path.join(validation_dir, 'daisy')
        self._val_dandelion_dir = os.path.join(validation_dir, 'dandelion')
        self._val_sunflowers_dir = os.path.join(validation_dir, 'sunflowers')
        self._val_tulips_dir = os.path.join(validation_dir, 'tulips')

        self._train_dir = train_dir
        self._val_dir = validation_dir

    def __repr__(self) -> str:
        return 'Flower()'

    def __str__(self) -> str:
        num_roses_tr, num_roses_vl = self.roses_samples()
        num_daisy_tr, num_daise_vl = self.daisy_samples()
        num_dandelion_tr, num_dandelion_vl = self.dandelion_samples()
        num_sunflowers_tr, num_sunflowers_vl = self.sunflowers_samples()
        num_tulips_tr, num_tulips_vl = self.tulips_samples()

        return f'DogVSCata datasets with \n' \
               f'   ROSES      ---- TRAIN = {num_roses_tr} VAL = {num_roses_vl} \n' \
               f'   DAISY      ---- TRAIN = {num_daisy_tr} VAL = {num_daise_vl} \n' \
               f'   DANDELION  ---- TRAIN = {num_dandelion_tr} VAL = {num_dandelion_vl} \n' \
               f'   SUNFLOWERS ---- TRAIN = {num_sunflowers_tr} VAL = {num_sunflowers_vl} \n' \
               f'   TULIPS     ---- TRAIN = {num_tulips_tr} VAL = {num_tulips_vl} \n' \
               f'   TOTAL      ---- TRAIN = ' \
               f'{num_roses_tr + num_daisy_tr + num_dandelion_tr + num_sunflowers_tr + num_tulips_tr}' \
               f' VAL = {num_roses_vl + num_daise_vl + num_dandelion_vl + num_sunflowers_vl + num_tulips_vl}\n'

    def roses_samples(self) -> Tuple[int, int]:
        num_tr = len(os.listdir(self._train_roses_dir))
        num_vl = len(os.listdir(self._val_roses_dir))

        return num_tr, num_vl

    def daisy_samples(self) -> Tuple[int, int]:
        num_tr = len(os.listdir(self._train_daisy_dir))
        num_vl = len(os.listdir(self._val_daisy_dir))

        return num_tr, num_vl

    def dandelion_samples(self) -> Tuple[int, int]:
        num_tr = len(os.listdir(self._train_dandelion_dir))
        num_vl = len(os.listdir(self._val_dandelion_dir))

        return num_tr, num_vl

    def sunflowers_samples(self) -> Tuple[int, int]:
        num_tr = len(os.listdir(self._train_sunflowers_dir))
        num_vl = len(os.listdir(self._val_sunflowers_dir))

        return num_tr, num_vl

    def tulips_samples(self) -> Tuple[int, int]:
        num_tr = len(os.listdir(self._train_tulips_dir))
        num_vl = len(os.listdir(self._val_tulips_dir))

        return num_tr, num_vl

    def train_image_generator_flow(self, batch_size: int, target_size: int) -> preprocessing.image.DirectoryIterator:
        return Flower._image_genrator_flow(batch_size=batch_size,
                                           target_size=target_size,
                                           directory=self._train_dir,
                                           horizontal_flip=True,
                                           shear_range=0.15,
                                           width_rang=0.15,
                                           zoom_range=0.5,
                                           heigth_range=0.15,
                                           rotation_range=45)

    def validation_image_generator_flow(self, batch_size: int, target_size: int):
        return Flower._image_genrator_flow(batch_size=batch_size, target_size=target_size,
                                           directory=self._val_dir)

    @staticmethod
    def plots_images(images_arr: np.array):
        fig, axes = plt.subplots(1, len(images_arr), figsize=(20, 20))
        axes = axes.flatten()
        for img, ax in zip(images_arr, axes):
            ax.imshow(img)
        plt.tight_layout()
        plt.show()

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
                                 class_mode='sparse')


def _split_train_validation_set(base_path: str, labels: List[str]):
    for cl in labels:
        image_path = os.path.join(base_path, cl)
        images = glob.glob(os.path.join(image_path, '*.jpg'))
        train, val = images[:round(len(images) * 0.8)], images[round(len(images) * 0.8):]
        if not os.path.exists(os.path.join(base_path, 'train', cl)):
            os.makedirs((os.path.join(base_path, 'train', cl)))
        for t in train:
            if os.path.exists(os.path.join(base_path, 'train', cl, t)):
                continue
            shutil.move(t, os.path.join(base_path, 'train', cl))

        if not os.path.exists(os.path.join(base_path, 'val', cl)):
            os.makedirs((os.path.join(base_path, 'val', cl)))
        for v in val:
            if os.path.exists(os.path.join(base_path, 'train', cl, v)):
                continue
            shutil.move(v, os.path.join(base_path, 'val', cl))
