# This is the training file for the model. It contains the training steps necessary.
# 1. Load the uicrit dataset and the images from the file system via UIMate/src/data/
# 2. Create the dictionary for the ratings
# 3. Create the training data in the correct order
# 4. Shuffle and split the data into training and validation sets
# 5. Define the model
# 6. Compile and train the model
# 7. Evaluate the model
# 8. Save the model to the file system

import os
import sys
import time

import keras
import tensorflow as tf

# add the path to the training module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "train")))

from data_loader import (
    get_mean_ratings,
    load_data,
)
from loss import weighted_mse_around_mean
from model_factory import (
    create_simple_cnn,
    create_mean_baseline_model,
    create_pretrained_resnet_cnn,
    create_cnn_softmax_output,
    create_pretrained_resnet_cnn_with_features,
    create_resnet_cnn_softmax_output,
    create_simpler_cnn_softmax_output,
)
from eval import evaluate_model, evaluate_model_per_dimension
from logger import TrainingLogger

LOSS = os.getenv("LOSS", "mse")


@keras.saving.register_keras_serializable(package="Custom", name="custom_weighted_mse")
def custom_weighted_mse(y_true, y_pred):
    """Wraps the weighted MSE around mean in a Keras-serializable function."""
    mean_ratings = get_mean_ratings()
    return weighted_mse_around_mean(y_true, y_pred, mean_ratings, 0.5)


def define_model():
    """Defines and returns the model and base_name."""
    model, base_name = create_pretrained_resnet_cnn_with_features(
        input_shape=(896, 896, 3),
        edge_input_shape=(56, 56),
        histogram_input_shape=(512,),
        output_dim=5,
        trainable=False,
    )
    return model, base_name


def define_model_one_dimension(num_classes):
    """Defines and returns the model and base_name."""
    model, base_name = create_resnet_cnn_softmax_output(
        input_shape=(896, 896, 3), num_classes=num_classes
    )
    return model, base_name


def compile_model(model):
    """Compiles the model with the given loss and metrics."""
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss=custom_weighted_mse,
        metrics=["mae"],
    )


def compile_softmax_one_dimension_model(model, learning_rate=1e-4):
    """Compiles the model with the given loss and metrics."""
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return learning_rate


def train_model(
    model, X_train, y_train, X_val, y_val, epochs=10, batch_size=16, logger=None
):
    """Trains the model and returns the training history."""
    print("Starting training...")
    start_time = time.time()

    callbacks = []
    if logger:
        callbacks.append(logger.create_keras_callback())

    history = model.fit(
        x={
            "image_input": X_train["image_input"],
            "histogram_input": X_train["histogram_input"],
            "edge_input": X_train["edge_input"],
        },
        y=y_train,
        validation_data=(
            {
                "image_input": X_val["image_input"],
                "histogram_input": X_val["histogram_input"],
                "edge_input": X_val["edge_input"],
            },
            y_val,
        ),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
    )
    end_time = time.time()
    print(f"Training completed in {end_time - start_time:.2f} seconds.")
    return history


def train_model_one_dimension(
    model,
    dimension_name,
    train_ds,
    val_ds,
    num_classes,
    train_samples,
    val_samples,
    epochs=10,
    batch_size=16,
    logger=None,
):
    """Trains the model for one dimension and returns the training history."""
    print("Starting training for dimension:", dimension_name)

    steps_per_epoch = train_samples // batch_size
    validation_steps = val_samples // batch_size

    callbacks = []
    if logger:
        callbacks.append(logger.create_keras_callback())

    start_time = time.time()
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
        batch_size=batch_size,
        callbacks=callbacks,
    )
    end_time = time.time()
    print(f"Training completed in {end_time - start_time:.2f} seconds.")
    return history


def save_model(model, base_name, dimension_name=None, logger=None):
    """Saves the model under the designated name and path."""
    model_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "model"))
    os.makedirs(model_dir, exist_ok=True)

    # create a timestamp for the model name
    timestamp = time.strftime("%Y%m%d-%H%M%S")

    # add dimension information to name if provided
    dimension_info = f"-{dimension_name}" if dimension_name else ""

    model_name = f"{base_name}{dimension_info}-{timestamp}-{LOSS}.keras"
    model_path = os.path.join(model_dir, model_name)
    model.save(model_path)
    print(f"Model saved to {model_path}.")

    # log the model path if logger is provided
    if logger:
        logger.log_model_path(model_path)

    return model_path


def evaluate_trained_model(model_path, X_val, y_val, logger=None):
    """Evaluates the trained model using the given validation data."""
    evaluate_model(model_path=model_path, X_val=X_val, y_val=y_val, logger=logger)


def main():
    # initialize the logger
    logger = TrainingLogger()

    # set the dimension name:
    dim_name = "usability_rating"

    # set max samples per class for undersampling (None for no undersampling)
    max_samples_per_class = 500

    # load the initial data with undersampling for balanced classes
    train_ds, val_ds, uicrit, rating_scales, train_samples, val_samples = load_data(
        dimension_name=dim_name, max_samples_per_class=max_samples_per_class
    )

    num_classes = rating_scales[dim_name]
    print(f"Training model for dimension {dim_name} with {num_classes} classes.")

    # 5. Define the model
    model, base_name = define_model_one_dimension(num_classes)

    # 6. Compile the model
    learning_rate = 1e-4
    learning_rate = compile_softmax_one_dimension_model(model, learning_rate)

    logger.log_config(
        model_name=base_name,
        dimension_name=dim_name,
        num_classes=num_classes,
        batch_size=32,
        learning_rate=learning_rate,
        epochs=15,
        loss_function="categorical_crossentropy",
        additional_info={
            "train_samples": train_samples,
            "val_samples": val_samples,
            "undersampling": (
                f"{max_samples_per_class} samples per class"
                if max_samples_per_class
                else "None"
            ),
        },
    )

    # 7. Train the model
    history = train_model_one_dimension(
        model,
        dim_name,
        train_ds,
        val_ds,
        num_classes,
        train_samples,
        val_samples,
        epochs=15,
        batch_size=32,
        logger=logger,
    )

    # 8. Save the model
    model_path = save_model(model, base_name, dimension_name=dim_name, logger=logger)

    # 9. Evaluate the model and log results
    eval_metrics = evaluate_model_per_dimension(
        model_path, dim_name, num_classes, val_ds, logger=logger
    )

    # finalize logging
    logger.finalize()


if __name__ == "__main__":
    main()
