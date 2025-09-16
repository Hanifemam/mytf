import tensorflow as tf


class Conv2D:
    def __init__(self, in_channels, out_channels, kernel_size, padding=0, stride=1):
        kh = kw = kernel_size
        fan_in = kh * kw * in_channels
        stddev = tf.sqrt(2.0 / tf.cast(fan_in, tf.float32))
        self.weight = tf.Variable(
            tf.random.normal([kh, kw, in_channels, out_channels], stddev=stddev),
            trainable=True,
        )
        self.bias = tf.Variable(tf.zeros([out_channels]), trainable=True)
        self.padding = int(padding)
        self.stride = int(stride)

    def __call__(self, x):
        # pad NHWC: [[N],[H],[W],[C]]
        p = self.padding
        if p:
            x = tf.pad(
                x,
                paddings=[[0, 0], [p, p], [p, p], [0, 0]],
                mode="CONSTANT",
                constant_values=0,
            )

        # shapes after padding
        shp = tf.shape(x)
        N = shp[0]
        Hp = shp[1]
        Wp = shp[2]
        Cin = shp[3]

        # kernel dims / Cout
        wshp = tf.shape(self.weight)
        kh = wshp[0]
        kw = wshp[1]
        Cout = wshp[3]

        s = self.stride

        # output sizes (VALID conv on padded input)
        Hout = (Hp - kh) // s + 1
        Wout = (Wp - kw) // s + 1

        y = tf.TensorArray(dtype=x.dtype, size=N)

        for n in tf.range(N):
            row_buf = tf.TensorArray(dtype=x.dtype, size=Hout)
            for i_out in tf.range(Hout):
                i0 = i_out * s
                col_buf = tf.TensorArray(dtype=x.dtype, size=Wout)
                for j_out in tf.range(Wout):
                    j0 = j_out * s
                    # patch: [kh, kw, Cin]
                    patch = x[n, i0 : i0 + kh, j0 : j0 + kw, :]
                    # sum over (kh,kw,Cin) -> [Cout]
                    val = (
                        tf.tensordot(patch, self.weight, axes=[[0, 1, 2], [0, 1, 2]])
                        + self.bias
                    )
                    col_buf = col_buf.write(j_out, val)
                row_buf = row_buf.write(
                    i_out, tf.stack(col_buf.stack(), axis=0)
                )  # [Wout, Cout]
            y = y.write(n, tf.stack(row_buf.stack(), axis=0))  # [Hout, Wout, Cout]
        return y.stack()  # [N, Hout, Wout, Cout]


# ------------------ tiny test (matches built-in) ------------------
x = tf.constant(
    [[[[1.0, 10.0], [2.0, 20.0]], [[3.0, 30.0], [4.0, 40.0]]]], dtype=tf.float32
)  # [1,2,2,2]

# Kernel: [2, 2, Cin=2, Cout=1]
w = tf.constant(
    [[[[1.0], [100.0]], [[2.0], [200.0]]], [[[3.0], [300.0]], [[4.0], [400.0]]]],
    dtype=tf.float32,
)
b = tf.zeros([1], dtype=tf.float32)

# Built-in conv (VALID, stride=1)
y_tf = tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding="VALID") + b

# Manual conv (stride=1, padding=0)
manual = Conv2D(in_channels=2, out_channels=1, kernel_size=2, padding=0, stride=1)
manual.weight.assign(w)
manual.bias.assign(b)
y_manual = manual(x)

print("Built-in TF:", y_tf.numpy())  # expected [[[[30030.]]]]
print("Manual TF: ", y_manual.numpy())  # should match
print("Max abs diff:", float(tf.reduce_max(tf.abs(y_tf - y_manual))))

# Optional: show stride=2 also works (output will be same shape here since kh==H==W==2)
y_tf_s2 = tf.nn.conv2d(x, w, strides=[1, 2, 2, 1], padding="VALID") + b
manual_s2 = Conv2D(in_channels=2, out_channels=1, kernel_size=2, padding=0, stride=2)
manual_s2.weight.assign(w)
manual_s2.bias.assign(b)
y_manual_s2 = manual_s2(x)
print("Stride=2 max diff:", float(tf.reduce_max(tf.abs(y_tf_s2 - y_manual_s2))))
