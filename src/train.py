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
import time

from sklearn.metrics import mean_absolute_error, mean_squared_error
import tensorflow as tf

from data_loader import (
    get_training_data,
    get_validation_data,
    get_evaluation_split_size,
)

from model_factory import create_simple_cnn, build_mean_baseline_model


LOSS = os.getenv("LOSS", "mse")

# Get the training data:
X_train, y_train = get_training_data()

# Get the validation data:
X_val, y_val = get_validation_data()

# 5. Define the model

model, base_name = build_mean_baseline_model(input_shape=(224, 224, 3))

# 6. Compile and train the model
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), loss="mse", metrics=["mae"]
)

print("Starting training...")
start_time = time.time()
history = model.fit(
    X_train, y_train, validation_data=(X_val, y_val), epochs=15, batch_size=16
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
