import argparse
from argparse import Namespace
import numpy as np
from tensorflow.keras import layers
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.regularizers import l2
from utils import *


class VGG(Model):
    def __init__(self, args: Namespace):
        super(VGG, self).__init__()
        # initialize model
        self.model: Sequential = Sequential()
        # features
        VGG._make_layers(self.model, args)
        # flatten layer
        self.model.add(layers.Flatten())
        # classifier head
        VGG._make_fc(self.model, args)

    def call(self, x):
        # print(x.shape)
        x = self.model(x)
        # print(x.shape)
        return x

    @staticmethod
    def _make_fc(classifier, args):
        conv_width: int = args.conv_width
        dropout: float = args.dropout
        num_block: int = args.num_block
        num_dense: int = args.num_dense
        weight_decay: float = args.weight_decay

        classifier.add(layers.Dropout(rate=dropout))
        if num_dense == 1:
            # only other option currently is to offer 2 dense layers
            classifier.add(layers.Dense(10))
        elif conv_width == 64:
            num_flat_features = 4096
            if num_block == 4:
                num_flat_features = 256
            classifier.add(layers.Dense(num_flat_features, kernel_regularizer=l2(weight_decay)))
            classifier.add(layers.Activation('relu'))
            classifier.add(layers.Dropout(rate=dropout))
            classifier.add(layers.Dense(10))
        elif conv_width == 128:
            num_flat_features = 8192
            if num_block == 4:
                num_flat_features = 512
            classifier.add(layers.Dense(num_flat_features, kernel_regularizer=l2(weight_decay)))
            classifier.add(layers.Activation('relu'))
            classifier.add(layers.Dropout(rate=dropout))
            classifier.add(layers.Dense(10))
        elif conv_width == 256:
            num_flat_features = 16384
            if num_block == 4:
                num_flat_features = 1024
            classifier.add(layers.Dense(num_flat_features, kernel_regularizer=l2(weight_decay)))
            classifier.add(layers.Activation('relu'))
            classifier.add(layers.Dropout(rate=dropout))
            classifier.add(layers.Dense(10))
        elif conv_width == 512:
            num_flat_features = 32768
            if num_block == 4:
                num_flat_features = 2048
            classifier.add(layers.Dense(num_flat_features, kernel_regularizer=l2(weight_decay)))
            classifier.add(layers.Activation('relu'))
            classifier.add(layers.Dropout(rate=dropout))
            classifier.add(layers.Dense(10))

    @staticmethod
    def _make_layers(features, args):
        conv_width: int = args.conv_width
        num_block: int = args.num_block
        weight_decay: float = args.weight_decay

        # First Block
        features.add(layers.Conv2D(conv_width, (3, 3), padding='same',
                                   input_shape=[32, 32, 3],
                                   kernel_regularizer=l2(weight_decay)))
        features.add(layers.Activation('relu'))
        # features.add(layers.Dropout(dropout)) we shouldn't have dropout in conv block
        features.add(layers.Conv2D(conv_width, (3, 3), padding='same', kernel_regularizer=l2(weight_decay)))
        features.add(layers.Activation('relu'))
        features.add(layers.MaxPooling2D(pool_size=(2, 2)))

        for _ in range(num_block-1):
            features.add(layers.Conv2D(conv_width, (3, 3), padding='same',
                                       kernel_regularizer=l2(weight_decay)))
            features.add(layers.Activation('relu'))
            # features.add(layers.Dropout(dropout))
            features.add(layers.Conv2D(conv_width, (3, 3), padding='same',
                                       kernel_regularizer=l2(weight_decay)))
            features.add(layers.Activation('relu'))
            features.add(layers.MaxPooling2D(pool_size=(2, 2)))
        return features


def _test():
    parser = argparse.ArgumentParser()
    parser.add_argument('--conv_width', type=int, default=128)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--num_block', type=int, default=4)
    parser.add_argument('--num_dense', type=int, default=2)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    hparams = parser.parse_args()
    net = VGG(hparams)

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    x_train, x_test = normalize(x_train, x_test)
    train_loader = tf.data.Dataset.from_tensor_slices((x_train, y_train)) \
        .map(prepare_data).shuffle(50000).batch(50000) #.batch(hparams.batch_size)
    # net = VGG16((32, 32, 3))
    import time
    for (data, _) in train_loader:
        print(data.shape)
        start = time.time()
        net(data)
        end = time.time()
        print(end - start)

    # x = np.random.randn(500, 32, 32, 3)


if __name__ == '__main__':
    _test()
