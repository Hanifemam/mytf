import tensorflow as tf


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

    def __init__(self, input_size, output_size, activation="ReLU"):
        """
        Initializes the Linear layer with random weights and biases.

        Args:
            input_size (int): Number of input features.
            output_size (int): Number of output features.
        """
        self.input_size = input_size
        self.output_size = output_size
        self._W = tf.Variable(
            tf.random.normal((input_size, output_size)),
            dtype=tf.float32,
            trainable=True,
            name="weights",
        )
        self._b = tf.Variable(
            tf.random.normal((output_size,)),
            dtype=tf.float32,
            trainable=True,
            name="bias",
        )

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
        return tf.matmul(x, self._W) + self._b

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


# Test the Linear layer implementation
if __name__ == "__main__":
    layer = Linear(input_size=3, output_size=2, activation="Sigmoid")

    x = tf.random.normal((4, 3))
    output = layer(x)

    print("Input:\n", x.numpy())
    print("Output:\n", output.numpy())
    print("Weights:\n", layer.W.numpy())
    print("Weights shape:", layer.W.shape)
    print("Biases:\n", layer.b.numpy())
