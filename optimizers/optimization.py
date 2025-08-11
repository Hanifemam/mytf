import tensorflow as tf
import sys
import os
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from losses.loss_functions import (
    MSELoss,
    MAELoss,
    BCELoss,
    CrossEntropyLoss,
    SparseCategoricalCrossEntropy,
)
from layers.sequence import Sequence
import layers.activations
from layers.feedforward import Linear


class Optimization:
    """
    Optimization class for training a model using SGD.
    """

    def __init__(
        self,
        model,
        X,
        y,
        epochs,
        loss_function,
        learning_rate,
        learning_rate_type="constant",
    ):
        self.model = model
        self.X = X
        self.y = y
        self.epochs = epochs
        self.loss_function = loss_function
        self.learning_rate = learning_rate
        self.learning_rate_type = learning_rate_type

    def SGD(self):
        """
        Apply Stochastic Gradient Descent to minimize the loss.

        Returns:
            list: The final model parameters (weights and biases).
        """
        momentum = 0.9
        accumulators = []
        # Fix params list once so velocities line up with params
        momentum_velocities = [tf.zeros_like(p) for p in list(self.model.parameters())]

        if self.learning_rate_type == "linear":
            lr_schedule = self.linear_lr_schedule(total_steps=self.epochs)

        for i in range(self.epochs):
            with tf.GradientTape() as tape:
                prediction = self.model(self.X)
                loss = self.loss_function(prediction, self.y)

            parameters = self.model.parameters()
            grads = tape.gradient(loss, parameters)
            if self.learning_rate_type == "constant":
                for param, grad in zip(parameters, grads):
                    if grad is None:
                        continue
                    param.assign_sub(self.learning_rate * grad)
            elif self.learning_rate_type == "momentum":
                for idx, (param, grad) in enumerate(zip(parameters, grads)):
                    if grad is None:
                        continue
                    momentum_velocities[idx] = (
                        momentum * momentum_velocities[idx] - self.learning_rate * grad
                    )
                    param.assign_add(momentum_velocities[idx])
            elif self.learning_rate_type == "linear":
                lr_t = float(lr_schedule[i])
                for param, grad in zip(parameters, grads):
                    if grad is None:
                        continue
                    param.assign_sub(lr_t * grad)

            elif self.learning_rate_type == "adagrad":
                if i == 0:
                    init_acc = 0.1
                    accumulators = [
                        tf.Variable(tf.fill(tf.shape(p), init_acc), trainable=False)
                        for p in parameters
                    ]
                else:
                    self.adagrad(grads, accumulators)
                for idx, (param, grad) in enumerate(zip(parameters, grads)):
                    if grad is None:
                        continue
                    lr = self.learning_rate / (tf.sqrt(accumulators[idx]) + 1e-8)
                    param.assign_sub(lr * grad)
            elif self.learning_rate_type == "RMSProp":
                if i == 0:
                    init_acc = 0.1
                    accumulators = [
                        tf.Variable(
                            tf.fill(tf.shape(p), tf.cast(init_acc, p.dtype)),
                            trainable=False,
                        )
                        for p in parameters
                    ]
                else:
                    self.RMSProp(grads, accumulators)
                for idx, (param, grad) in enumerate(zip(parameters, grads)):
                    if grad is None:
                        continue
                    lr = self.learning_rate / (tf.sqrt(accumulators[idx]) + 1e-8)
                    param.assign_sub(lr * grad)
            elif self.learning_rate_type == "Adam":
                if i == 0:
                    init_acc = 0.0
                    momentum_velocities = [
                        tf.Variable(
                            tf.fill(tf.shape(p), init_acc),
                            trainable=False,
                            dtype=tf.float32,
                        )
                        for p in parameters
                    ]

                    accumulators = [
                        tf.Variable(
                            tf.fill(tf.shape(p), init_acc),
                            trainable=False,
                            dtype=tf.float32,
                        )
                        for p in parameters
                    ]

                    lr_correction = self.Adam(
                        grads,
                        accumulators,
                        momentum_velocities,
                        tf.cast(i + 1, tf.float32),
                    )
                else:
                    lr_correction = self.Adam(
                        grads,
                        accumulators,
                        momentum_velocities,
                        tf.cast(i + 1, tf.float32),
                    )
                for param, corr in zip(parameters, lr_correction):
                    if corr is None:
                        continue
                    param.assign_sub(corr * self.learning_rate)

            else:
                raise ValueError(
                    f"Unknown learning_rate_type: {self.learning_rate_type}"
                )

            print(f"Epoch {i+1}: Loss = {loss.numpy():.4f}")

        return self.model.parameters()

    def linear_lr_schedule(
        self, total_steps, base_lr=1e-3, end_frac=0.01, warmup_ratio=0.05
    ):
        # Robust guards
        total_steps = max(int(total_steps), 1)
        warmup_steps = max(int(round(total_steps * warmup_ratio)), 0)
        min_lr = base_lr * end_frac
        decay_steps = max(total_steps - warmup_steps, 1)

        lrs = []
        for t in range(total_steps):
            if warmup_steps > 0 and t < warmup_steps:
                # Linear warmup 0 -> base_lr (use t+1 to avoid 0 when desired)
                lr = base_lr * ((t + 1) / warmup_steps)
            else:
                # Linear decay base_lr -> min_lr
                progress = (t - warmup_steps) / decay_steps
                progress = min(max(progress, 0.0), 1.0)
                lr = base_lr - progress * (base_lr - min_lr)
            lrs.append(lr)
        return np.array(lrs, dtype=np.float32)

    def power_law_lr_schedule(self):
        pass

    def exponential_lr_schedule(self):
        pass

    def adagrad(self, grads, accumulators):
        for i, g in enumerate(grads):
            if g is None:
                continue
            accumulators[i].assign_add(tf.square(g))  # persistent, in-place
        return accumulators

    def RMSProp(self, grads, accumulators, decay_rate=0.9):
        for i, g in enumerate(grads):
            if g is None:
                continue
            accumulators[i] = decay_rate * accumulators[i] + (
                1 - decay_rate
            ) * tf.square(g)
        return accumulators

    def Adam(
        self,
        grads,
        accumulators,
        momentum_velocities,
        time_round,  # t (use i+1 when calling)
        decay_rate_momentum=0.9,  # β1
        decay_rate_suare=0.999,  # β2  (kept your name, value fixed)
        delta=1e-8,
    ):
        learning_rate_correction = []
        for i, g in enumerate(grads):
            if g is None:
                learning_rate_correction.append(None)
                continue
            momentum_velocities[i].assign(
                decay_rate_momentum * momentum_velocities[i]
                + (1 - decay_rate_momentum) * g
            )

            accumulators[i].assign(
                decay_rate_suare * accumulators[i]
                + (1.0 - decay_rate_suare) * tf.square(g)
            )
            m_hat = momentum_velocities[i] / (
                1.0 - tf.pow(decay_rate_momentum, time_round)
            )
            v_hat = accumulators[i] / (1.0 - tf.pow(decay_rate_suare, time_round))
            learning_rate_correction.append(m_hat / (tf.sqrt(v_hat) + delta))

        return learning_rate_correction


X = tf.random.normal((10, 3))
y = tf.random.normal((10, 1))

model = Sequence(Linear(3, 4), Linear(4, 1))

optimizer = Optimization(
    model=model,
    X=X,
    y=y,
    epochs=10,
    loss_function=MSELoss,
    learning_rate=0.01,
    learning_rate_type="Adam",
)
final_params = optimizer.SGD()

print("\nFinal Parameters:")
for param in final_params:
    print(param.numpy())
