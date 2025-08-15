import tensorflow as tf
import numpy as np

import tensorflow as tf


class Normalization(tf.keras.layers.Layer):
    def __init__(self, epsilon=1e-7, name=None):
        super().__init__(name=name)
        self.epsilon = epsilon

    def call(self, x):
        x = tf.cast(x, tf.float32)
        mean = tf.reduce_mean(x, axis=0, keepdims=True)
        std = tf.math.reduce_std(x, axis=0, keepdims=True)
        return (x - mean) / (std + self.epsilon)


if __name__ == "__main__":
    import numpy as np

    data = np.random.randn(5, 3) * 5 + 20  # 5 samples, 3 features
    print("Unnormalized data: ", data)
    norm_layer = Normalization()
    normed = norm_layer(data).numpy()
    print("Means:", normed.mean(axis=0))
    print("Stds :", normed.std(axis=0, ddof=0))
    print("Normalized data: ", normed)
