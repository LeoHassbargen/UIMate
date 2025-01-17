import tensorflow as tf


def weighted_mse_around_mean(y_true, y_pred, mean_vals, alpha=1.0):
    """
    Weighted MSE, das Datenpunkte, die weiter vom Mittelwert entfernt sind,
    stärker gewichtet. 'mean_vals' sollte ein Tensor mit den globalen Mittelwerten
    der jeweiligen Dimension sein, z.B. aus df.describe().
    alpha steuert die Stärke der Gewichtung.

    y_true, y_pred: (batch_size, n_dims)
    mean_vals: (n_dims,) -> z.B. [mean_dim0, mean_dim1, ...]
    """

    # Abweichung vom Mittelwert je Dimension
    # shape: (batch_size, n_dims)
    dist_from_mean = tf.abs(y_true - mean_vals)

    # Ermitteln des Gewichts je nach Entfernung vom Mittelwert
    # z.B. w = 1 + alpha * dist_from_mean
    w = 1.0 + alpha * dist_from_mean

    # MSE = mean( w * (y_true - y_pred)^2 )
    squared_diff = tf.square(y_true - y_pred)
    weighted_loss = tf.reduce_mean(w * squared_diff)
    return weighted_loss
