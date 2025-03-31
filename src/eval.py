import os
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    explained_variance_score,
)
import tensorflow as tf
import numpy as np
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix

import matplotlib.pyplot as plt
import seaborn as sns


def evaluate_model(model_path, X_val, y_val, logger=None):
    # Load the model
    model = tf.keras.models.load_model(model_path)

    # Make the predictions
    predictions = model.predict(X_val)

    # Print the variance of the predictions
    predictions_variance = float(np.var(predictions))
    true_values_variance = float(np.var(y_val))
    print(f"Variance of the predictions: {predictions_variance:.4f}")
    print(f"Variance of the true values: {true_values_variance:.4f}")

    # Compute the overall metrics
    mse = float(mean_squared_error(y_val, predictions))
    mae = float(mean_absolute_error(y_val, predictions))
    evs = float(explained_variance_score(y_val, predictions))

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

    # Create a dict to store evaluation results
    eval_metrics = {
        "overall_mse": mse,
        "overall_mae": mae,
        "overall_evs": evs,
        "predictions_variance": predictions_variance,
        "true_values_variance": true_values_variance,
        "dimensions": {},
    }

    # 2) Loop through each dimension to compute metrics individually
    print("\nDetailed per-dimension metrics:")
    for i, rating_name in enumerate(rating_scales.keys()):
        dim_mse = float(mean_squared_error(y_val[:, i], predictions[:, i]))
        dim_mae = float(mean_absolute_error(y_val[:, i], predictions[:, i]))
        dim_evs = float(explained_variance_score(y_val[:, i], predictions[:, i]))

        eval_metrics["dimensions"][rating_name] = {
            "mse": dim_mse,
            "mae": dim_mae,
            "evs": dim_evs,
        }

        print(
            f"  {rating_name}: MSE = {dim_mse:.4f}, MAE = {dim_mae:.4f}, EVS = {dim_evs:.4f}"
        )


def evaluate_model_per_dimension(
    model_path, dimension_name, dimension_classes, val_ds, logger=None
):
    """Evaluates the trained model for a specific dimension with a softmax output.

    The model output is assumed to be a probability distribution over `dimension_classes` classes.
    Precision, Recall, F1-Score, and Accuracy are computed based on the predicted and true labels.

    Args:
        model_path: Path to the saved model
        dimension_name: Name of the dimension being evaluated
        dimension_classes: Number of classes in the dimension
        val_ds: Validation dataset (tf.data.Dataset)
        logger: Optional TrainingLogger instance

    Returns:
        Dictionary containing evaluation metrics
    """
    # load the model
    model = tf.keras.models.load_model(model_path)

    # get the softmax predictions for the specified dimension
    predictions = []
    y_true = []
    for X_batch, y_batch in val_ds:
        preds = model.predict(X_batch)
        predictions.append(preds)
        y_true.append(y_batch)

    predictions = np.concatenate(predictions, axis=0)
    y_true = np.concatenate(y_true, axis=0)

    # convert softmax probabilities to predicted class labels
    y_pred = np.argmax(predictions, axis=1)

    # if y_val is one-hot encoded, convert it to class labels.
    if y_true.ndim > 1 and y_true.shape[1] == dimension_classes:
        y_true = np.argmax(y_true, axis=1)

    # compute Precision, Recall, F1-Score, and Accuracy
    precision = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    recall = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    accuracy = accuracy_score(y_true, y_pred)

    # compute the confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # create metrics dictionary for logging
    metrics = {
        "dimension": dimension_name,
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "accuracy": float(accuracy),
        "num_classes": dimension_classes,
        "class_metrics": {},
    }

    # per-class metrics (precision, recall, f1 for each class)
    for i in range(dimension_classes):
        class_metrics = {
            "precision": float(
                precision_score(
                    y_true, y_pred, labels=[i], average="micro", zero_division=0
                )
            ),
            "recall": float(
                recall_score(
                    y_true, y_pred, labels=[i], average="micro", zero_division=0
                )
            ),
            "f1": float(
                f1_score(y_true, y_pred, labels=[i], average="micro", zero_division=0)
            ),
            "support": int(np.sum(y_true == i)),
        }
        metrics["class_metrics"][f"class_{i}"] = class_metrics

    # distribution of predictions
    pred_distribution = {
        int(i): int(np.sum(y_pred == i)) for i in range(dimension_classes)
    }
    metrics["prediction_distribution"] = pred_distribution

    # true label distribution
    true_distribution = {
        int(i): int(np.sum(y_true == i)) for i in range(dimension_classes)
    }
    metrics["true_distribution"] = true_distribution

    # print the metrics
    print(f"Metrics for dimension '{dimension_name}':")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1-Score: {f1:.4f}")
    print(f"  Accuracy: {accuracy:.4f}")

    # print the confusion matrix
    print(f"Confusion Matrix:\n{cm}")

    # if logger is provided, log the evaluation results
    if logger:
        plt.figure(figsize=(12, 10))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=range(dimension_classes),
            yticklabels=range(dimension_classes),
        )
        plt.title(f"Confusion Matrix - {dimension_name}")
        plt.xlabel("Predicted label")
        plt.ylabel("True label")

        cm_plot_path = os.path.join(
            logger.plots_dir, f"confusion_matrix_{dimension_name}.png"
        )
        plt.savefig(cm_plot_path)
        plt.close()

    return metrics
