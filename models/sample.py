import numpy as np
from tensorflow.keras.layers import Activation, Conv2D, Dense, Flatten
from tensorflow.keras.models import Model, Sequential


class Network(Model):
    "use for testing"
    def __init__(self):
        super(Network, self).__init__()
        model = Sequential()
        model.add(Conv2D(16, (9, 9), input_shape=(32, 32, 3)))
        model.add(Activation('relu'))
        model.add(Conv2D(16, (3, 3)))
        model.add(Flatten())
        model.add(Dense(10))
        self.model = model

    def call(self, x):
        x = self.model(x)
        return x


if __name__ == '__main__':
    net = Network()
    x = np.random.randn(500, 32, 32, 3)
    out = net(x)
