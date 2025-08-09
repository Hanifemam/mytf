import tensorflow as tf
import sys
import os

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

        # Fix params list once so velocities line up with params
        params = list(self.model.parameters())
        momentum_velocities = [tf.zeros_like(p) for p in params]
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
                for idx, (param, grad) in enumerate(zip(params, grads)):
                    if grad is None:
                        continue
                    momentum_velocities[idx] = (
                        momentum * momentum_velocities[idx] - self.learning_rate * grad
                    )
                    param.assign_add(momentum_velocities[idx])
            else:
                raise ValueError(
                    f"Unknown learning_rate_type: {self.learning_rate_type}"
                )

            print(f"Epoch {i+1}: Loss = {loss.numpy():.4f}")

        return self.model.parameters()


X = tf.random.normal((10, 3))
y = tf.random.normal((10, 1))

model = Sequence(Linear(3, 4), Linear(4, 1))

optimizer = Optimization(
    model=model, X=X, y=y, epochs=5, loss_function=MSELoss, learning_rate=0.01
)

final_params = optimizer.SGD()

print("\nFinal Parameters:")
for param in final_params:
    print(param.numpy())
