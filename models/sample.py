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
        self.model = model
        self.flatten = Flatten()
        self.fc = Dense(10)

    def call(self, x):
        x = self.model(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x


if __name__ == '__main__':
    net = Network()
    x = np.random.randn(500, 32, 32, 3)
    out = net(x)
