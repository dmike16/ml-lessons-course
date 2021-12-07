# -----------------------------------------------------------
# Class to manage the MNIST dataset
#
#
# The model is created usgin tensorflow + + tensorflow_dataset  keras + numpy
#
# dmike16, Rome, Italy
# Released under MIT license
# email cipmiky@gmail.com
# -----------------------------------------------------------

"""
Module containing utility class to prepare plot, tain and validate the MNIST
fashion ML problem
"""

from typing import List
import tensorflow as tf
import tensorflow_datasets as tfds


class FashionMNIST:
    @property
    def ds_train(self) -> tf.data.Dataset:
        return self._train

    @property
    def ds_test(self) -> tf.data.Dataset:
        return self._test

    @property
    def info(self):
        return self._info

    def class_names(self)-> List:
        return self.info.features['label'].names

    def __init__(self) -> None:
        (ds_train, ds_test), metainfo = tfds.load('fashion_mnist', as_supervised=True, with_info=True)
        self._info = metainfo
        self._train = ds_train
        self._test = ds_test

    def __repr__(self) -> str:
        return f'FashionMNIST()'

    def __str__(self) -> str:
        meta_train = self.info.splits['train']
        meta_test = self.info.splits['test']
        return f'FashionMNIST data set with {meta_train.num_examples} train and {meta_test.num_examples} test'
