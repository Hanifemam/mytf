import tensorflow as tf


class Conv2D:

    def __init__(self, in_channels, out_channels, kernel_size, padding=0, stride=0):
        kh = kw = kernel_size
        fan_in = kh * kw * in_channels
        stddev = tf.math.sqrt(2.0 / fan_in)
        self.weight = tf.Variable(
            tf.random.normal([kh, kw, in_channels, out_channels], stddev=stddev),
            trainable=True,
        )
        self.bias = tf.Variable(tf.zeros([out_channels]))
        self.padding = padding
        self.stride = stride

    def __call__(self, x):
        N, H, W, Cin = tf.shape(x)
        kh, kw, _, cout = self.weight.shape

        Hout = H - kh + 1
        Wout = W - kw + 1

        y = tf.TensorArray(dtype=x.dtype, size=N)

        for n in range(N):
            row_buf = tf.TensorArray(dtype=x.dtype, size=Hout)
            for i in range(Hout):
                col_buf = tf.TensorArray(dtype=x.dtype, size=Wout)
                for j in range(Wout):
                    patch = x[n, i : i + kh, j : j + kw, :]
                    val = (
                        tf.tensordot(patch, self.weight, axes=[[0, 1, 2], [0, 1, 2]])
                        + self.bias
                    )
                    col_buf = col_buf.write(j, val)
                row_buf = row_buf.write(
                    i, tf.stack(col_buf.stack(), axis=0)
                )  # [Wout, Cout]
            y = y.write(n, tf.stack(row_buf.stack(), axis=0))
        return y.stack()


x = tf.constant(
    [[[[1.0, 10.0], [2.0, 20.0]], [[3.0, 30.0], [4.0, 40.0]]]], dtype=tf.float32
)

# Kernel: [2, 2, Cin=2, Cout=1]
w = tf.constant(
    [[[[1.0], [100.0]], [[2.0], [200.0]]], [[[3.0], [300.0]], [[4.0], [400.0]]]],
    dtype=tf.float32,
)
b = tf.zeros([1], dtype=tf.float32)

# Built-in conv (VALID, stride=1)
y_tf = tf.nn.conv2d(x, w, strides=1, padding="VALID") + b

# Manual conv with identical weights/bias
manual = Conv2D(in_channels=2, out_channels=1, kernel_size=2)
manual.weight.assign(w)
manual.bias.assign(b)

y_manual = manual(x)

print("Built-in TF:", y_tf.numpy())  # expected [[[[30030.]]]]
print("Manual TF: ", y_manual.numpy())  # should match
print("Max abs diff:", tf.reduce_max(tf.abs(y_tf - y_manual)).numpy())
