import os
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    explained_variance_score,
)
import tensorflow as tf
import numpy as np


def evaluate_model(model_path, X_val, y_val):
    # Load the model
    model = tf.keras.models.load_model(model_path)

    # Make the predictions
    predictions = model.predict(X_val)

    # Print the variance of the predictions
    print(f"Variance of the predictions: {np.var(predictions):.4f}")
    print(f"Variance of the true values: {np.var(y_val):.4f}")

    # Compute the overall metrics
    mse = mean_squared_error(y_val, predictions)
    mae = mean_absolute_error(y_val, predictions)
    evs = explained_variance_score(y_val, predictions)

    print(f"Overall Test MSE: {mse:.4f}")
    print(f"Overall Test MAE: {mae:.4f}")
    print(f"Overall Test EVS: {evs:.4f}")

    # 1) Define rating names for the 5 predicted dimensions:
    rating_scales = np.load(
        os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "data", "rating_scales.npy")
        ),
        allow_pickle=True,
    ).item()

    # 2) Loop through each dimension to compute metrics individually
    print("\nDetailed per-dimension metrics:")
    per_dimension_metrics = {}

    for i, rating_name in enumerate(rating_scales.keys()):
        dim_mse = mean_squared_error(y_val[:, i], predictions[:, i])
        dim_mae = mean_absolute_error(y_val[:, i], predictions[:, i])
        dim_evs = explained_variance_score(y_val[:, i], predictions[:, i])

        print(
            f"  {rating_name}: MSE = {dim_mse:.4f}, MAE = {dim_mae:.4f}, EVS = {dim_evs:.4f}"
        )
