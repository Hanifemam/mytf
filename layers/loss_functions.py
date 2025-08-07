"""This module contains custom loss functions for training neural networks.
loss functions for neural networks."""

import tensorflow as tf


def MSELoss(predictions, targets):
    """
    Computes the Mean Squared Error (MSE) loss between predictions and targets.

    Args:
        predictions (tf.Tensor): Predicted values from the model.
        targets (tf.Tensor): Ground truth values.

    Returns:
        tf.Tensor: Computed MSE loss.
    """
    return tf.reduce_mean(tf.square(predictions - targets))


def MAELoss(predictions, targets):
    """
    Computes the Mean Absolute Error (MAE) loss between predictions and targets.

    Args:
        predictions (tf.Tensor): Predicted values from the model.
        targets (tf.Tensor): Ground truth values.

    Returns:
        tf.Tensor: Computed MAE loss.
    """
    return tf.reduce_mean(tf.abs(predictions - targets))


def BCELoss(predictions, targets, epsilon=1e-7):
    """
    Binary Cross Entropy loss.

    Args:
        predictions (tf.Tensor): Predicted probabilities (after sigmoid).
        targets (tf.Tensor): Ground truth labels (0 or 1).
        epsilon (float): Small constant for numerical stability.

    Returns:
        tf.Tensor: Scalar BCE loss.
    """
    predictions = tf.clip_by_value(predictions, epsilon, 1 - epsilon)  # avoid log(0)
    return tf.reduce_mean(
        -(
            targets * tf.math.log(predictions)
            + (1 - targets) * tf.math.log(1 - predictions)
        )
    )


def CrossEntropyLoss(predictions, targets, epsilon=1e-7):
    """
    Cross Entropy loss for multi-class classification.

    Args:
        predictions (tf.Tensor): Predicted probabilities (after softmax).
        targets (tf.Tensor): One-hot encoded ground truth labels.
        epsilon (float): Small constant for numerical stability.

    Returns:
        tf.Tensor: Scalar cross entropy loss.
    """
    predictions = tf.clip_by_value(predictions, epsilon, 1 - epsilon)
    return tf.reduce_mean(tf.reduce_sum(-targets * tf.math.log(predictions), axis=1))


import tensorflow as tf


def SparseCategoricalCrossEntropy(predictions, targets, epsilon=1e-7):
    """
    Computes the Sparse Categorical Cross Entropy loss.

    Args:
        predictions (tf.Tensor): Predicted probabilities, shape (batch_size, num_classes).
        targets (tf.Tensor): Integer class labels, shape (batch_size,).
        epsilon (float): Small constant to avoid log(0).

    Returns:
        tf.Tensor: Scalar tensor representing the average cross entropy loss.
    """
    # Ensure predictions are clipped to avoid log(0)
    predictions = tf.clip_by_value(predictions, epsilon, 1 - epsilon)

    # Convert class indices to log-probabilities at the correct index
    # predictions[batch_index, target_class]
    batch_indices = tf.range(tf.shape(targets)[0])
    indices = tf.stack([batch_indices, targets], axis=1)  # shape: (batch_size, 2)

    # Gather the correct class probabilities
    true_probs = tf.gather_nd(predictions, indices)  # shape: (batch_size,)

    # Compute -log(p)
    loss = -tf.math.log(true_probs)

    # Return average loss
    return tf.reduce_mean(loss)


# # === 1. MSELoss (regression)
# print("=== MSELoss ===")
# preds = tf.constant([[0.5], [0.2], [0.9]], dtype=tf.float32)
# targets = tf.constant([[1.0], [0.0], [1.0]], dtype=tf.float32)
# print("MSE Loss:", MSELoss(preds, targets).numpy())

# # === 2. MAELoss (regression)
# print("\n=== MAELoss ===")
# print("MAE Loss:", MAELoss(preds, targets).numpy())

# # === 3. BCELoss (binary classification)
# print("\n=== BCELoss ===")
# preds_bce = tf.constant([[0.9], [0.2], [0.6]], dtype=tf.float32)
# targets_bce = tf.constant([[1.0], [0.0], [1.0]], dtype=tf.float32)
# print("BCE Loss:", BCELoss(preds_bce, targets_bce).numpy())

# # === 4. CrossEntropyLoss (multi-class, one-hot labels)
# print("\n=== CrossEntropyLoss ===")
# preds_ce = tf.constant([[0.1, 0.7, 0.2], [0.8, 0.1, 0.1]], dtype=tf.float32)
# targets_ce = tf.constant([[0, 1, 0], [1, 0, 0]], dtype=tf.float32)
# print("Cross Entropy Loss:", CrossEntropyLoss(preds_ce, targets_ce).numpy())

# # === 5. SparseCategoricalCrossEntropy (multi-class, integer labels)
# print("\n=== SparseCategoricalCrossEntropy ===")
# targets_sparse = tf.constant(
#     [1, 0], dtype=tf.int32
# )  # same labels as above, but not one-hot
# print(
#     "Sparse Cross Entropy Loss:",
#     SparseCategoricalCrossEntropy(preds_ce, targets_sparse).numpy(),
# )
