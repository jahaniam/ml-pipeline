import tensorflow as tf


class MnistDataloader:
    def __init__(self):
        mnist = tf.keras.datasets.mnist

        (self.x_train, self.y_train), (self.x_test, self.y_test) = mnist.load_data()
        self.x_train, self.x_test = self.x_train / 255.0, self.x_test / 255.0

        # Add a channels dimension
        self.x_train = self.x_train[..., tf.newaxis].astype("float32")
        self.x_test = self.x_test[..., tf.newaxis].astype("float32")

    def get_train_ds(self, batch_size=32):
        train_ds = tf.data.Dataset.from_tensor_slices((self.x_train, self.y_train)).shuffle(10000).batch(batch_size)
        # TODO: add augmentation train_ds = train_ds.map(...)

        return train_ds

    def get_test_ds(self, batch_size=32):
        test_ds = tf.data.Dataset.from_tensor_slices((self.x_test, self.y_test)).batch(batch_size)
        return test_ds
