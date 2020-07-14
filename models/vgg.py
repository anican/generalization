import argparse
from argparse import Namespace
import numpy as np
from tensorflow.keras import layers
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.regularizers import l2


class VGG(Model):
    def __init__(self, args: Namespace):
        super(VGG, self).__init__()
        num_dense: int = args.num_dense
        self.features = VGG._make_layers(args)
        self.flatten = layers.Flatten()
        self.classifier = VGG._make_fc(args)

    def call(self, x):
        print(x.shape)
        x = self.features(x)
        x = self.flatten(x)
        x = self.classifier(x)
        print(x.shape)
        return x

    @staticmethod
    def _make_fc(args):
        # TODO: currently only supports block size of 2
        classifier = Sequential()
        conv_width: int = args.conv_width
        num_dense: int = args.num_dense
        weight_decay: float = args.weight_decay
        if conv_width == 64:
            if num_dense == 1:
                classifier.add(layers.Dense(10))
            else:
                classifier.add(layers.Dense(4096, kernel_regularizer=l2(weight_decay)))
                classifier.add(layers.Activation('relu'))
                classifier.add(layers.Dense(10))
        elif conv_width == 256:
            if num_dense == 1:
                classifier.add(layers.Dense(10))
            else:
                classifier.add(layers.Dense(16384, kernel_regularizer=l2(weight_decay)))
                classifier.add(layers.Activation('relu'))
                classifier.add(layers.Dense(10))
        return classifier

    @staticmethod
    def _make_layers(args):
        features = Sequential()
        conv_width: int = args.conv_width
        num_block: int = args.num_block
        weight_decay: float = args.weight_decay
        dropout: float = args.dropout

        # First Block
        features.add(layers.Conv2D(conv_width, (3, 3), padding='same',
                                   input_shape=[32, 32, 3],
                                   kernel_regularizer=l2(weight_decay)))
        features.add(layers.Activation('relu'))
        features.add(layers.Dropout(dropout))
        features.add(layers.Conv2D(conv_width, (3, 3), padding='same', kernel_regularizer=l2(weight_decay)))
        features.add(layers.Activation('relu'))
        features.add(layers.MaxPooling2D(pool_size=(2, 2)))

        for _ in range(num_block-1):
            features.add(layers.Conv2D(conv_width, (3, 3), padding='same',
                                       kernel_regularizer=l2(weight_decay)))
            features.add(layers.Activation('relu'))
            features.add(layers.Dropout(dropout))
            features.add(layers.Conv2D(conv_width, (3, 3), padding='same',
                                       kernel_regularizer=l2(weight_decay)))
            features.add(layers.Activation('relu'))
            features.add(layers.MaxPooling2D(pool_size=(2, 2)))
        return features


def _test():
    parser = argparse.ArgumentParser()
    parser.add_argument('--conv_width', type=int, default=64)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--num_block', type=int, default=2)
    parser.add_argument('--num_dense', type=int, default=2)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    hparams = parser.parse_args()
    net = VGG(hparams)
    print('net', net)
    # net = VGG16((32, 32, 3))
    x = np.random.randn(500, 32, 32, 3)
    out = net(x)


if __name__ == '__main__':
    _test()
