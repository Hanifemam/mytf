"""
activations.py

Contains activation functions used in neural networks.
Each activation function is implemented as a callable class,
allowing consistent usage with layer-like objects.
"""

import tensorflow as tf

class ReLU:
    """Applies the ReLU activation function."""

    def __call__(self, x):
        return tf.maximum(0.0, x)

    def __repr__(self):
        return "ReLU()"


class LeakyReLU:
    """Applies the Leaky ReLU activation function."""

    def __init__(self, alpha=0.2):
        self.alpha = alpha

    def __call__(self, x):
        return tf.where(x >= 0, x, self.alpha * x)

    def __repr__(self):
        return f"LeakyReLU(alpha={self.alpha})"


class Sigmoid:
    """Applies the sigmoid activation function."""

    def __call__(self, x):
        return 1 / (1 + tf.exp(-x))

    def __repr__(self):
        return "Sigmoid()"


class Tanh:
    """Applies the hyperbolic tangent activation function."""

    def __call__(self, x):
        return tf.tanh(x)

    def __repr__(self):
        return "Tanh()"


class Softmax:
    """Applies the softmax activation function."""

    def __init__(self, axis=-1):
        self.axis = axis

    def __call__(self, x):
        return tf.nn.softmax(x, axis=self.axis)

    def __repr__(self):
        return f"Softmax(axis={self.axis})"

"""
activations.py

Contains activation functions used in neural networks.
Each activation function is implemented as a callable class,
allowing consistent usage with layer-like objects.
"""

import tensorflow as tf

class ReLU:
    """Applies the ReLU activation function."""

    def __call__(self, x):
        return tf.maximum(0.0, x)

    def __repr__(self):
        return "ReLU()"


class LeakyReLU:
    """Applies the Leaky ReLU activation function."""

    def __init__(self, alpha=0.2):
        self.alpha = alpha

    def __call__(self, x):
        return tf.where(x >= 0, x, self.alpha * x)

    def __repr__(self):
        return f"LeakyReLU(alpha={self.alpha})"


class Sigmoid:
    """Applies the sigmoid activation function."""

    def __call__(self, x):
        return 1 / (1 + tf.exp(-x))

    def __repr__(self):
        return "Sigmoid()"


class Tanh:
    """Applies the hyperbolic tangent activation function."""

    def __call__(self, x):
        return tf.tanh(x)

    def __repr__(self):
        return "Tanh()"


class Softmax:
    """Applies the softmax activation function."""

    def __init__(self, axis=-1):
        self.axis = axis

    def __call__(self, x):
        return tf.nn.softmax(x, axis=self.axis)

    def __repr__(self):
        return f"Softmax(axis={self.axis})"

