import tensorflow as tf
import numpy as np


class Linear:
    """
    A custom implementation of a fully connected (dense) linear layer.

    This layer performs the operation: output = x @ W + b,
    where `x` is the input tensor, `W` is the weight matrix,
    and `b` is the bias vector.

    Attributes:
        input_size (int): The number of input features.
        output_size (int): The number of output features.
        _W (tf.Variable): The weight matrix of shape (input_size, output_size).
        _b (tf.Variable): The bias vector of shape (output_size,).
    """

    def __init__(
        self, input_size, output_size, residual=False, dropout=0.5, training=True
    ):
        """
        Initializes the Linear layer with random weights and biases.

        Args:
            input_size (int): Number of input features.
            output_size (int): Number of output features.
        """
        self.input_size = input_size
        self.output_size = output_size
        mean = 0.0
        std = np.sqrt(2.0 / (input_size))
        mean = 0.0
        std = np.sqrt(2.0 / (input_size))
        self._W = tf.Variable(
            tf.random.normal((input_size, output_size), mean, std),
            dtype=tf.float32,
            trainable=True,
            name="weights",
        )
        self._b = tf.Variable(
            tf.random.normal((output_size,), mean, std),
            dtype=tf.float32,
            trainable=True,
            name="bias",
        )
        self.residual = residual
        self.p_keep = dropout
        self.training = training

    def __call__(self, x):
        """
        Enables the layer to be called like a function.

        Args:
            x (tf.Tensor or np.ndarray): Input tensor.

        Returns:
            tf.Tensor: Output after applying the linear transformation.
        """
        return self.forward(x)

    def __repr__(self):
        """
        Returns a string representation of the Linear layer.

        Returns:
            str: A human-readable description of the layer.
        """
        return f"Linear(input_size={self.input_size}, output_size={self.output_size})"

    def forward(self, x):
        """
        Forward pass of the Linear layer.

        Args:
            x (tf.Tensor or np.ndarray): Input data of shape (batch_size, input_size).

        Returns:
            tf.Tensor: Output of shape (batch_size, output_size).
        """
        if not isinstance(x, tf.Tensor):
            x = tf.convert_to_tensor(x, dtype=tf.float32)
        y = tf.matmul(x, self._W) + self._b  # [batch, out_features]

        # Residual: only valid if shapes match
        if self.residual:
            if x.shape != y.shape:
                raise ValueError(
                    f"Residual requires in==out, got {self.in_features}!={self.out_features}"
                )
            y = y + x
        # Inverted dropout (train only)
        if self.training and self.p_keep < 1.0:
            mask = tf.cast(tf.random.uniform(tf.shape(y)) < self.p_keep, y.dtype)
            y = y * mask / self.p_keep

        return y

    @property
    def parameters(self):
        """
        Returns the trainable parameters of the layer.

        Returns:
            list: A list containing the weight matrix and bias vector.
        """
        return [self._W, self._b]

    @property
    def W(self):
        """
        Accesses the weight matrix.

        Returns:
            tf.Variable: The weight matrix.
        """
        return self._W

    @W.setter
    def W(self, value):
        """
        Sets the weight matrix with a new value.

        Args:
            value (np.ndarray or tf.Tensor): New weight values.

        Raises:
            ValueError: If the input is not a valid array or tensor.
        """
        if not isinstance(value, tf.Tensor):
            value = tf.convert_to_tensor(value, dtype=tf.float32)
        self._W.assign(value)

    @property
    def b(self):
        """
        Accesses the bias vector.

        Returns:
            tf.Variable: The bias vector.
        """
        return self._b

    @b.setter
    def b(self, value):
        """
        Sets the bias vector with a new value.

        Args:
            value (np.ndarray or tf.Tensor): New bias values.

        Raises:
            ValueError: If the input is not a valid array or tensor.
        """
        if not isinstance(value, tf.Tensor):
            value = tf.convert_to_tensor(value, dtype=tf.float32)
        self._b.assign(value)


import tensorflow as tf
import numpy as np


class BatchNormalization:
    """
    Batch Normalization for 2D inputs shaped (N, F).
    Keeps your original API: _W (gamma), _b (beta), parameters, W/b properties.
    """

    def __init__(self, input_size, epsilon=1e-5, momentum=0.99):
        self.input_size = int(input_size)
        self.epsilon = float(epsilon)
        self.momentum = float(momentum)

        self._W = tf.Variable(
            tf.ones((self.input_size,), dtype=tf.float32),  # gamma
            trainable=True,
            name="gamma",
        )
        self._b = tf.Variable(
            tf.zeros((self.input_size,), dtype=tf.float32),  # beta
            trainable=True,
            name="beta",
        )

        self.moving_mean = tf.Variable(
            tf.zeros((self.input_size,), dtype=tf.float32),
            trainable=False,
            name="moving_mean",
        )
        self.moving_var = tf.Variable(
            tf.ones((self.input_size,), dtype=tf.float32),
            trainable=False,
            name="moving_var",
        )

    def __call__(self, x, training=True):
        """Apply BatchNorm.
        Args:
            x: (N, F) tensor
            training (bool): if True use batch stats and update running stats,
            else use moving stats.
        """
        x = tf.convert_to_tensor(x, dtype=tf.float32)

        if training:
            batch_mean = tf.reduce_mean(x, axis=0)
            batch_var = tf.math.reduce_variance(x, axis=0)

            self.moving_mean.assign(
                self.momentum * self.moving_mean + (1.0 - self.momentum) * batch_mean
            )
            self.moving_var.assign(
                self.momentum * self.moving_var + (1.0 - self.momentum) * batch_var
            )

            mean, var = batch_mean, batch_var
        else:
            mean, var = self.moving_mean, self.moving_var

        # Normalize with standard deviation (sqrt(variance)), not variance
        x_hat = (x - mean) / tf.sqrt(var + self.epsilon)  # (N, F)

        # Affine transform: gamma (W) and beta (b)
        return x_hat * self._W + self._b

    # ---- Compatibility with your original API ----
    @property
    def parameters(self):
        return [self._W, self._b]

    @property
    def W(self):  # gamma
        return self._W

    @W.setter
    def W(self, value):
        value = tf.convert_to_tensor(value, dtype=tf.float32)
        if value.shape != self._W.shape:
            raise ValueError(f"W shape {value.shape} != {self._W.shape}")
        self._W.assign(value)

    @property
    def b(self):  # beta
        return self._b

    @b.setter
    def b(self, value):
        value = tf.convert_to_tensor(value, dtype=tf.float32)
        if value.shape != self._b.shape:
            raise ValueError(f"b shape {value.shape} != {self._b.shape}")
        self._b.assign(value)


import tensorflow as tf


class LayerNormalization:
    """
    Layer Normalization for 2D inputs shaped (N, F).
    Keeps your original API: _W (gamma), _b (beta), parameters, W/b properties.
    """

    def __init__(self, input_size, epsilon=1e-5):
        self.input_size = int(input_size)
        self.epsilon = float(epsilon)

        # Learnable affine params (gamma, beta), shape (F,)
        self._W = tf.Variable(
            tf.ones((self.input_size,), dtype=tf.float32),  # gamma
            trainable=True,
            name="gamma",
        )
        self._b = tf.Variable(
            tf.zeros((self.input_size,), dtype=tf.float32),  # beta
            trainable=True,
            name="beta",
        )

    def __call__(self, x):
        """
        Apply LayerNorm across the feature dimension for each sample.
        Args:
            x: (N, F) tensor
        Returns:
            (N, F) tensor
        """
        x = tf.convert_to_tensor(x, dtype=tf.float32)

        # Per-sample stats across features
        mean = tf.reduce_mean(x, axis=-1, keepdims=True)  # (N, 1)
        var = tf.math.reduce_variance(x, axis=-1, keepdims=True)  # (N, 1)

        x_hat = (x - mean) / tf.sqrt(var + self.epsilon)  # (N, F)

        # Affine transform (broadcast (F,) over batch)
        return x_hat * self._W + self._b

    # ---- Compatibility with your original API ----
    @property
    def parameters(self):
        return [self._W, self._b]

    @property
    def W(self):  # gamma
        return self._W

    @W.setter
    def W(self, value):
        value = tf.convert_to_tensor(value, dtype=tf.float32)
        if value.shape != self._W.shape:
            raise ValueError(f"W shape {value.shape} != {self._W.shape}")
        self._W.assign(value)

    @property
    def b(self):  # beta
        return self._b

    @b.setter
    def b(self, value):
        value = tf.convert_to_tensor(value, dtype=tf.float32)
        if value.shape != self._b.shape:
            raise ValueError(f"b shape {value.shape} != {self._b.shape}")
        self._b.assign(value)


# Test the Linear layer implementation
if __name__ == "__main__":
    # layer = Linear(input_size=3, output_size=2, activation="Sigmoid")

    # x = tf.random.normal((4, 3))
    # output = layer(x)

    # print("Input:\n", x.numpy())
    # print("Output:\n", output.numpy())
    # print("Weights:\n", layer.W.numpy())
    # print("Weights shape:", layer.W.shape)
    # print("Biases:\n", layer.b.numpy())
    # Create a BatchNormalization layer for 3 features
    layer = LayerNormalization(input_size=3)

    # Random input: 4 samples, 3 features
    x = tf.random.normal(mean=5.0, stddev=2.0, shape=(4, 3))

    # Forward pass (training mode)
    output = layer(x)

    print("Input:\n", x.numpy())
    print("Output (normalized):\n", output.numpy())
    print("Gamma (W):\n", layer.W.numpy())
    print("Gamma shape:", layer.W.shape)
    print("Beta (b):\n", layer.b.numpy())
    print("Beta shape:", layer.b.shape)
