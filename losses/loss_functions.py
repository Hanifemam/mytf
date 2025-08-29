"""This module contains custom loss functions for training neural networks.
loss functions for neural networks."""

import tensorflow as tf


def MSELoss(
    predictions,
    targets,
    regularizer="l2",
    parameters=None,
    decay=1e-3,
):
    """
    Computes the Mean Squared Error (MSE) loss between predictions and targets.

    Args:
        predictions (tf.Tensor): Model predictions (any shape broadcastable to `targets`).
        targets (tf.Tensor): Ground-truth values (same shape as `predictions`).
        regularizer (str or None, optional): Regularizer type. Use "l2" to add an L2 penalty;
            use None or any other value to disable. Defaults to "l2".
        parameters (Sequence[tf.Variable | tf.Tensor] or tf.Variable or tf.Tensor or None, optional):
            Parameters to regularize. If None or empty, no regularization is applied even if
            `regularizer == "l2"`. Defaults to None.
        decay (float, optional): L2 coefficient λ applied as `λ * Σ‖W‖²`. Defaults to 1e-3.


    Returns:
        tf.Tensor: Computed MSE loss.
    """
    l2 = 0
    if regularizer == "l2" and parameters:
        l2 = l2_regularizer(parameters=parameters, decay=decay)
    return tf.reduce_mean(tf.square(predictions - targets)) + tf.convert_to_tensor(
        l2, dtype=predictions.dtype
    )


def MAELoss(
    predictions,
    targets,
    regularizer="l2",
    parameters=None,
    decay=1e-3,
):
    """
    Computes the Mean Absolute Error (MAE) loss between predictions and targets.

    Args:
        predictions (tf.Tensor): Model predictions (any shape broadcastable to `targets`).
        targets (tf.Tensor): Ground-truth values (same shape as `predictions`).
        regularizer (str or None, optional): Regularizer type. Use "l2" to add an L2 penalty;
            use None or any other value to disable. Defaults to "l2".
        parameters (Sequence[tf.Variable | tf.Tensor] or tf.Variable or tf.Tensor or None, optional):
            Parameters to regularize. If None or empty, no regularization is applied even if
            `regularizer == "l2"`. Defaults to None.
        decay (float, optional): L2 coefficient λ applied as `λ * Σ‖W‖²`. Defaults to 1e-3.


    Returns:
        tf.Tensor: Computed MAE loss.
    """
    l2 = 0
    if regularizer == "l2" and parameters:
        l2 = l2_regularizer(parameters=parameters, decay=decay)
    return tf.reduce_mean(tf.abs(predictions - targets)) + tf.convert_to_tensor(
        l2, dtype=predictions.dtype
    )


def BCELoss(
    predictions,
    targets,
    epsilon=1e-7,
    regularizer="l2",
    parameters=None,
    decay=1e-3,
):
    """
    Binary Cross Entropy loss.

    Args:
        predictions (tf.Tensor): Model predictions (any shape broadcastable to `targets`).
        targets (tf.Tensor): Ground-truth values (same shape as `predictions`).
        regularizer (str or None, optional): Regularizer type. Use "l2" to add an L2 penalty;
            use None or any other value to disable. Defaults to "l2".
        parameters (Sequence[tf.Variable | tf.Tensor] or tf.Variable or tf.Tensor or None, optional):
            Parameters to regularize. If None or empty, no regularization is applied even if
            `regularizer == "l2"`. Defaults to None.
        decay (float, optional): L2 coefficient λ applied as `λ * Σ‖W‖²`. Defaults to 1e-3.
        epsilon (float): Small constant for numerical stability.

    Returns:
        tf.Tensor: Scalar BCE loss.
    """
    l2 = 0
    if regularizer == "l2" and parameters:
        l2 = l2_regularizer(parameters=parameters, decay=decay)
    predictions = tf.clip_by_value(predictions, epsilon, 1 - epsilon)  # avoid log(0)
    return tf.reduce_mean(
        -(
            targets * tf.math.log(predictions)
            + (1 - targets) * tf.math.log(1 - predictions)
        )
    ) + tf.convert_to_tensor(l2, dtype=predictions.dtype)


def CrossEntropyLoss(
    predictions,
    targets,
    epsilon=1e-7,
    regularizer="l2",
    parameters=None,
    decay=1e-3,
):
    """
    Cross Entropy loss for multi-class classification.

    Args:
        predictions (tf.Tensor): Model predictions (any shape broadcastable to `targets`).
        targets (tf.Tensor): Ground-truth values (same shape as `predictions`).
        regularizer (str or None, optional): Regularizer type. Use "l2" to add an L2 penalty;
            use None or any other value to disable. Defaults to "l2".
        parameters (Sequence[tf.Variable | tf.Tensor] or tf.Variable or tf.Tensor or None, optional):
            Parameters to regularize. If None or empty, no regularization is applied even if
            `regularizer == "l2"`. Defaults to None.
        decay (float, optional): L2 coefficient λ applied as `λ * Σ‖W‖²`. Defaults to 1e-3.

        epsilon (float): Small constant for numerical stability.

    Returns:
        tf.Tensor: Scalar cross entropy loss.
    """
    l2 = 0
    if regularizer == "l2" and parameters:
        l2 = l2_regularizer(parameters=parameters, decay=decay)
    predictions = tf.clip_by_value(predictions, epsilon, 1 - epsilon)
    return tf.reduce_mean(
        tf.reduce_sum(-targets * tf.math.log(predictions), axis=1)
    ) + tf.convert_to_tensor(l2, dtype=predictions.dtype)


def SparseCategoricalCrossEntropy(
    predictions,
    targets,
    epsilon=1e-7,
    regularizer="l2",
    parameters=None,
    decay=1e-3,
):
    """
    Computes the Sparse Categorical Cross Entropy loss.

    Args:
        predictions (tf.Tensor): Model predictions (any shape broadcastable to `targets`).
        targets (tf.Tensor): Ground-truth values (same shape as `predictions`).
        regularizer (str or None, optional): Regularizer type. Use "l2" to add an L2 penalty;
            use None or any other value to disable. Defaults to "l2".
        parameters (Sequence[tf.Variable | tf.Tensor] or tf.Variable or tf.Tensor or None, optional):
            Parameters to regularize. If None or empty, no regularization is applied even if
            `regularizer == "l2"`. Defaults to None.
        decay (float, optional): L2 coefficient λ applied as `λ * Σ‖W‖²`. Defaults to 1e-3.

        epsilon (float): Small constant to avoid log(0).

    Returns:
        tf.Tensor: Scalar tensor representing the average cross entropy loss.
    """
    l2 = 0
    predictions = tf.clip_by_value(predictions, epsilon, 1 - epsilon)
    batch_indices = tf.range(tf.shape(targets)[0])
    indices = tf.stack([batch_indices, targets], axis=1)
    true_probs = tf.gather_nd(predictions, indices)
    loss = -tf.math.log(true_probs)
    if regularizer == "l2" and parameters:
        l2 = l2_regularizer(parameters=parameters, decay=decay)
    return tf.reduce_mean(loss) + tf.convert_to_tensor(l2, dtype=predictions.dtype)


def l2_regularizer(parameters, decay=1e-3):
    """
    Computes an L2 penalty term over a set of parameters.

    Args:
        parameters (Sequence[tf.Variable | tf.Tensor] or tf.Variable or tf.Tensor):
            Parameters to regularize. Must be non-empty. For convenience, a single
            tensor/variable is also accepted.
        decay (float, optional): L2 coefficient λ. The returned value is
            `λ * Σ_i reduce_sum(square(p_i))`. Defaults to 1e-3.

    Returns:
        tf.Tensor: Scalar tensor representing the L2 penalty.

    Notes:
        This helper does not guard against an empty list; callers should pass a non-empty
        sequence or check before calling (as done in the loss functions above).
    """
    if parameters is None or decay == 0.0:
        return 0.0
    if isinstance(parameters, (tf.Tensor, tf.Variable)):
        params = [parameters]
    else:
        params = list(parameters)
    if len(params) == 0:
        return 0.0
    return decay * tf.add_n([tf.reduce_sum(tf.square(v)) for v in params])


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
if __name__ == "__main__":
    # Tiny smoke test: compare MSE with/without L2
    W = tf.Variable([[1.0, -2.0], [3.0, 0.5]], dtype=tf.float32)
    params = [W]
    decay = 1e-2

    preds = tf.constant([[0.5], [0.2], [0.9]], dtype=tf.float32)
    targs = tf.constant([[1.0], [0.0], [1.0]], dtype=tf.float32)

    no_reg = MSELoss(preds, targs, regularizer=None)
    with_l2 = MSELoss(preds, targs, parameters=params, decay=decay)

    print(f"MSE (no reg): {float(no_reg):.6f}")
    print(f"MSE (+L2):    {float(with_l2):.6f}")
