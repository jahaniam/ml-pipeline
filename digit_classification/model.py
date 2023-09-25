from __future__ import print_function

import logging

from tensorflow.keras import Model
from tensorflow.keras.layers import Concatenate, Conv2D, Dense, Flatten, MaxPooling2D

logging.basicConfig(level=logging.DEBUG)

# Define the model object


class MyModel(Model):
    def __init__(self):
        super().__init__()
        self.conv1 = Conv2D(32, 3, activation="relu")
        self.conv2_1 = Conv2D(64, 3, activation="relu")
        self.conv2_2 = Conv2D(64, 3, activation="relu")
        self.conv3_1 = Conv2D(256, 3, activation="relu")
        self.conv3_2 = Conv2D(256, 3, activation="relu")
        self.down_sample = MaxPooling2D()
        self.flatten = Flatten()
        self.d1 = Dense(1000, activation="relu")
        self.d2 = Dense(500, activation="relu")
        self.out = Dense(10, activation="softmax")

    def call(self, x):
        conv1_out = self.down_sample(self.conv1(x))

        conv2_1_out = self.down_sample(self.conv2_1(conv1_out))
        conv2_2_out = self.down_sample(self.conv2_2(conv1_out))

        conv3_1_out = self.down_sample(self.conv3_1(conv2_1_out))
        conv3_2_out = self.down_sample(self.conv3_2(conv2_2_out))

        conv3_out = Concatenate()([conv3_1_out, conv3_2_out])

        x = self.flatten(conv3_out)
        x = self.d1(x)
        x = self.d2(x)
        return self.out(x)
