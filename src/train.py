# This is the training file for the model. It contains the training steps necessary.
# 1. Load the uicrit dataset and the images from the file system via UIMate/src/data/
# 2. Create the dictionary for the ratings
# 3. Create the training data in the correct order
# 4. Shuffle and split the data into training and validation sets
# 5. Define the model
# 6. Compile and train the model
# 7. Evaluate the model
# 8. Save the model to the file system

import datetime
import json
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

# Fügen Sie den Pfad zu src/train hinzu
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "train")))
from model_factory import (
    create_simple_cnn,
    create_resnet_50_model,
    create_resnet_151v2_model,
)

# Import weighted_mse_around_mean from src/train/loss.py
from loss import weighted_mse_around_mean

from model_factory import (
    create_simple_cnn,
    create_mean_baseline_model,
    create_pretrained_resnet_cnn,
    create_pretrained_resnet_cnn_with_features,
)


LOSS = os.getenv("LOSS", "mse")
EPOCHS = os.getenv("EPOCHS", 10)

# Get the training data:
X_train, y_train = get_training_data()

# Get the validation data:
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
# Collect the duration using 2 digits after the comma.
train_duration = round(end_time - start_time, 2)
print(f"Training completed in {train_duration} seconds.")

# Logging
train_log = {
    "timestamp": datetime.datetime.now().isoformat(),
    "model_name": base_name,
    "loss_used": LOSS,
    "epochs": EPOCHS,
    "train_size": len(X_train),
    "val_size": len(X_val),
    "train_duration_seconds": train_duration,
    "history": {
        "loss": history.history["loss"],
        "val_loss": history.history["val_loss"],
        "mae": history.history["mae"],
        "val_mae": history.history["val_mae"],
    },
}

# 7. Save the model
model_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "model"))
os.makedirs(model_dir, exist_ok=True)

model_name = (
    base_name
    + "-"
    + str(get_evaluation_split_size())
    + "-"
    + LOSS
    + "-"
    + str(EPOCHS)
    + ".keras"
)

model_path = os.path.join(model_dir, model_name)
model.save(model_path)

print(f"Model saved to {model_path}.")

# 8. Evaluate the model
from eval import evaluate_model

evaluations = evaluate_model(model_path=model_path, X_val=X_val, y_val=y_val)

# Append the evalution results to the training log
train_log["evaluation"] = evaluations

# Save the training log
logs_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "logs"))
os.makedirs(logs_dir, exist_ok=True)
log_file = os.path.join(logs_dir, "training_logs.json")

# Lesen, anfügen, schreiben
logs_data = []
if os.path.exists(log_file):
    with open(log_file, "r") as f:
        try:
            logs_data = json.load(f)
        except json.JSONDecodeError:
            logs_data = []
logs_data.append(train_log)

with open(log_file, "w") as f:
    json.dump(logs_data, f, indent=2)
print(f"Training log appended to {log_file}.")
