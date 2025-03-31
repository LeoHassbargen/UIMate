import os
import json
import time
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Any, List, Optional, Tuple


class TrainingLogger:
    """
    A logger class for tracking training and evaluation metrics.
    Creates a unique directory for each run and saves metrics, plots, and configuration.
    """

    def __init__(self, base_dir: str = None):
        """Initialize the logger with a base directory."""
        if base_dir is None:
            base_dir = os.path.abspath(
                os.path.join(os.path.dirname(__file__), "..", "logs")
            )

        # Create base logs directory if it doesn't exist
        os.makedirs(base_dir, exist_ok=True)

        # Create a unique run directory with timestamp
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.run_dir = os.path.join(base_dir, f"run_{timestamp}")
        os.makedirs(self.run_dir, exist_ok=True)

        # Create subdirectories for different log types
        self.plots_dir = os.path.join(self.run_dir, "plots")
        self.models_dir = os.path.join(self.run_dir, "models")
        os.makedirs(self.plots_dir, exist_ok=True)
        os.makedirs(self.models_dir, exist_ok=True)

        # Initialize metrics tracking
        self.metrics = {"training": [], "evaluation": {}}

        self.config = {"timestamp": timestamp, "start_time": time.time()}

        print(f"Logger initialized. Logs will be saved to: {self.run_dir}")

    def log_config(
        self,
        model_name: str,
        dimension_name: str = None,
        num_classes: int = None,
        batch_size: int = None,
        learning_rate: float = None,
        epochs: int = None,
        loss_function: str = None,
        additional_info: Dict[str, Any] = None,
    ) -> None:
        """
        Log configuration information about the training run.

        Args:
            model_name: Name of the model architecture
            dimension_name: Name of the dimension being predicted
            num_classes: Number of classes in the classification task
            batch_size: Batch size used for training
            learning_rate: Learning rate for optimization
            epochs: Number of training epochs
            loss_function: Loss function used
            additional_info: Any additional information to log
        """
        self.config.update(
            {
                "model_name": model_name,
                "dimension_name": dimension_name,
                "num_classes": num_classes,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "epochs": epochs,
                "loss_function": loss_function,
            }
        )

        if additional_info:
            self.config.update(additional_info)

        # save the config to a JSON file
        config_path = os.path.join(self.run_dir, "config.json")
        with open(config_path, "w") as f:
            json.dump(self.config, f, indent=4)

        print(f"Configuration logged to: {config_path}")

    def log_epoch(self, epoch: int, logs: Dict[str, float]) -> None:
        """
        Log metrics for a single training epoch.

        Args:
            epoch: The epoch number
            logs: Dictionary containing metrics like loss, accuracy, etc.
        """
        epoch_data = {"epoch": epoch}
        epoch_data.update(logs)
        self.metrics["training"].append(epoch_data)

        # save training metrics to CSV after each epoch
        metrics_df = pd.DataFrame(self.metrics["training"])
        metrics_df.to_csv(
            os.path.join(self.run_dir, "training_metrics.csv"), index=False
        )

    def log_evaluation_results(
        self, metrics: Dict[str, Any], confusion_matrix: Optional[np.ndarray] = None
    ) -> None:
        """
        Log evaluation results.

        Args:
            metrics: Dictionary containing evaluation metrics
            confusion_matrix: Optional confusion matrix array
        """
        self.metrics["evaluation"] = metrics

        # save evaluation metrics to JSON
        eval_path = os.path.join(self.run_dir, "evaluation_results.json")
        with open(eval_path, "w") as f:
            json.dump(metrics, f, indent=4)

        # ff a confusion matrix is provided, save it as a numpy array
        if confusion_matrix is not None:
            np.save(
                os.path.join(self.run_dir, "confusion_matrix.npy"), confusion_matrix
            )

        print(f"Evaluation results logged to: {eval_path}")

    def create_training_plots(self) -> None:
        """Create and save plots for training metrics."""
        if not self.metrics["training"]:
            print("No training metrics to plot.")
            return

        df = pd.DataFrame(self.metrics["training"])
        metrics_to_plot = [col for col in df.columns if col != "epoch"]

        for metric in metrics_to_plot:
            plt.figure(figsize=(10, 6))
            plt.plot(df["epoch"], df[metric], label=f"Training {metric}")

            val_metric = f"val_{metric}"
            if val_metric in df.columns:
                plt.plot(df["epoch"], df[val_metric], label=f"Validation {metric}")

            plt.xlabel("Epoch")
            plt.ylabel(metric.capitalize())
            plt.title(f"{metric.capitalize()} over epochs")
            plt.legend()
            plt.grid(True)

            # save plot
            plot_path = os.path.join(self.plots_dir, f"{metric}_plot.png")
            plt.savefig(plot_path)
            plt.close()

        print(f"Training plots saved to: {self.plots_dir}")

    def log_model_path(self, model_path: str) -> str:
        """
        Log the saved model path and copy or create a symlink to the model in the logs directory.

        Args:
            model_path: Path where the model was saved

        Returns:
            Path to the model in the logs directory
        """
        self.config["model_path"] = model_path

        # update the config file with the model path
        config_path = os.path.join(self.run_dir, "config.json")
        with open(config_path, "w") as f:
            json.dump(self.config, f, indent=4)

        return model_path

    def finalize(self) -> None:
        """Finalize logging, create plots, and save any remaining data."""
        # calculate elapsed time
        elapsed_time = time.time() - self.config["start_time"]
        self.config["elapsed_time"] = elapsed_time

        # update the config file
        config_path = os.path.join(self.run_dir, "config.json")
        with open(config_path, "w") as f:
            json.dump(self.config, f, indent=4)

        # create training plots
        self.create_training_plots()

        print(f"Training completed in {elapsed_time:.2f} seconds.")
        print(f"All logs saved to: {self.run_dir}")

    def create_keras_callback(self) -> "KerasLoggerCallback":
        """
        Create a Keras callback that logs metrics during training.

        Returns:
            Keras callback for logging
        """
        from tensorflow.keras.callbacks import Callback

        logger = self

        class KerasLoggerCallback(Callback):
            def on_epoch_end(self, epoch, logs=None):
                logs = logs or {}
                logger.log_epoch(epoch, logs)

        return KerasLoggerCallback()
