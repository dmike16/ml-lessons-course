# -----------------------------------------------------------
# Class to manage the AUTO MPG dataset
#
#
# The model is created usgin tensorflow + pandas  keras + numpy
#
# dmike16, Rome, Italy
# Released under MIT license
# email cipmiky@gmail.com
# -----------------------------------------------------------

"""
Module containing utility class to prepare plot, train and validate the AUTO MPG
ML problem
"""

import pandas as pd
import tensorflow as tf
from tensorflow import keras
from typing import Tuple

_AUTOMPG_URL = 'http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data'
_COLUMNS_NAME = ['MPG', 'Cylinders', 'Displacement', 'Horsepower', 'Weight',
                 'Acceleration', 'Model Year', 'Origin']


class AutoMPG:
    def __init__(self):
        csv_file = keras.utils.get_file('autompg.csv', _AUTOMPG_URL)
        raw_dataset = pd.read_csv(csv_file, names=_COLUMNS_NAME, na_values='?', comment='\t', sep=' ',
                                  skipinitialspace=True)
        self.dataset = raw_dataset.copy()

    def tail(self):
        print(self.dataset.tail())

    def checl_for_unknow_values(self):
        print(self.dataset.isna().sum())

    def cleanup_data(self):
        """Cleanup and format the data
            1. remove unknow value
            2. transform categorical column 'Origin' in a numeric one
        """
        self.dataset = self.dataset.dropna()
        self.dataset['Origin'] = self.dataset['Origin'].map({1: 'USA', 2: 'Europe', 3: 'Japan'})
        self.dataset = pd.get_dummies(self.dataset, columns=['Origin'], prefix='', prefix_sep='')

    def split(self, fraction: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
        train = self.dataset.sample(frac=fraction, random_state=0)
        test = self.dataset.drop(train.index)
        return train, test

    @staticmethod
    def to_dataset(train: pd.DataFrame, test: pd.DataFrame) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
        """Transoform dataframe pandas to tensorflow dataset"""
        train_features = train.copy()
        test_features = test.copy()
        train_label = train_features.pop('MPG')
        test_label = test_features.pop('MPG')
        return tf.data.Dataset.from_tensor_slices((train_features, train_label)), tf.data.Dataset.from_tensor_slices(
            (test_features, test_label))

    @staticmethod
    def statistics(data: pd.DataFrame):
        print(data.describe().transpose())
