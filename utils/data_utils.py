"""
Utilities specific to dataset curation/manipulation
"""
import numpy as np
import tensorflow as tf


def prepare_data(x, y):
    x = tf.cast(x, tf.float32)
    y = tf.cast(y, tf.int32)
    return x, y


def normalize(x_train, x_test):
    # this function normalize inputs for zero mean and unit variance
    # it is used when training a model.
    # Input: training set and test set
    # Output: normalized training set and test set according to the train set statistics.
    x_train = x_train / 255.
    x_test = x_test / 255.
    mean = np.mean(x_train, axis=(0, 1, 2, 3))
    std = np.std(x_train, axis=(0, 1, 2, 3))
    print('mean:', mean, 'std:', std)
    x_train = (x_train - mean) / (std + 1e-7)
    x_test = (x_test - mean) / (std + 1e-7)
    return x_train, x_test
