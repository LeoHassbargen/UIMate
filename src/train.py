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

# Add the path to the training module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "train")))

from data_loader import (
    get_training_data,
    get_validation_data,
    get_evaluation_split_size,
    get_mean_ratings,
)
from loss import weighted_mse_around_mean
from model_factory import (
    create_simple_cnn,
    create_mean_baseline_model,
    create_pretrained_resnet_cnn,
    create_pretrained_resnet_cnn_with_features,
)
from eval import evaluate_model

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


def compile_model(model):
    """Compiles the model with the given loss and metrics."""
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss=custom_weighted_mse,
        metrics=["mae"],
    )


def train_model(model, X_train, y_train, X_val, y_val, epochs=10, batch_size=16):
    """Trains the model and returns the training history."""
    print("Starting training...")
    start_time = time.time()
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
    )
    end_time = time.time()
    print(f"Training completed in {end_time - start_time:.2f} seconds.")
    return history


def save_model(model, base_name):
    """Saves the model under the designated name and path."""
    model_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "model"))
    os.makedirs(model_dir, exist_ok=True)

    model_name = (
        base_name + "-" + str(get_evaluation_split_size()) + "-" + LOSS + ".keras"
    )
    model_path = os.path.join(model_dir, model_name)
    model.save(model_path)
    print(f"Model saved to {model_path}.")
    return model_path


def evaluate_trained_model(model_path, X_val, y_val):
    """Evaluates the trained model using the given validation data."""
    evaluate_model(model_path=model_path, X_val=X_val, y_val=y_val)


def main():
    # 1. Load the training data
    X_train, y_train = get_training_data()

    # 2. Load the validation data
    X_val, y_val = get_validation_data()

    # 5. Define the model
    model, base_name = define_model()

    # 6. Compile the model
    compile_model(model)

    # 7. Train the model
    train_model(model, X_train, y_train, X_val, y_val, epochs=1, batch_size=32)

    # 8. Save the model
    model_path = save_model(model, base_name)

    # 9. Evaluate the model
    evaluate_trained_model(model_path, X_val, y_val)


if __name__ == "__main__":
    main()
