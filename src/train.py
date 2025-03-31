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

from data_loader import (
    get_training_data,
    get_validation_data,
    get_evaluation_split_size,
    get_mean_ratings,
)

# add the path to the src directory to the sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "train")))

# import weighted_mse_around_mean from src/train/loss.py
from loss import weighted_mse_around_mean

from model_factory import (
    create_simple_cnn,
    create_mean_baseline_model,
    create_pretrained_resnet_cnn,
    create_pretrained_resnet_cnn_with_features,
)


LOSS = os.getenv("LOSS", "mse")

# get the training data:
X_train, y_train = get_training_data()

# get the validation data:
X_val, y_val = get_validation_data()

# 5. Define the model

model, base_name = create_pretrained_resnet_cnn_with_features(
    input_shape=(224, 224, 3),
    edge_input_shape=(56, 56),
    histogram_input_shape=(512,),
    output_dim=5,
    trainable=False,
)


@keras.saving.register_keras_serializable(package="Custom", name="custom_weighted_mse")
def custom_weighted_mse(y_true, y_pred):
    mean_ratings = get_mean_ratings()
    return weighted_mse_around_mean(y_true, y_pred, mean_ratings, 0.5)


# 6. Compile and train the model
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss=custom_weighted_mse,
    metrics=["mae"],
)

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
    epochs=10,
    batch_size=16,
)
end_time = time.time()
print(f"Training completed in {end_time - start_time:.2f} seconds.")


# 7. Save the model
model_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "model"))
os.makedirs(model_dir, exist_ok=True)

model_name = base_name + "-" + str(get_evaluation_split_size()) + "-" + LOSS + ".keras"

model_path = os.path.join(model_dir, model_name)
model.save(model_path)

print(f"Model saved to {model_path}.")

# 8. Evaluate the model
from eval import evaluate_model

evaluate_model(model_path=model_path, X_val=X_val, y_val=y_val)
