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
import pandas as pd
import time
import numpy as np

from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
import tensorflow as tf
from keras import models, layers

# 1. Load the data

try:
    images_dict = np.load(
        os.path.abspath(
            os.path.join(os.path.dirname(__file__), '..', 'data', 'images.npy')
        ),
        allow_pickle=True
    ).item()


    uicrit = pd.read_csv(
        os.path.abspath(
            os.path.join(os.path.dirname(__file__), '..', 'data', 'uicrit.csv')
        )
    )
except FileNotFoundError:
    print("Error: Could not find the images or uicrit dataset.")
    exit(1)

print(f"Loaded {len(images_dict)} images and {len(uicrit)} entries from the file system.")

# 2. Create the dictionary for the ratings
label_dict = {
    row['rico_id']: {
        'aesthetics_rating': row['aesthetics_rating'],
        'learnability': row['learnability'],
        'efficency': row['efficency'],
        'usability_rating': row['usability_rating'],
        'design_quality_rating': row['design_quality_rating']
    }
    for _, row in uicrit.iterrows()
}

# select all keys
all_ids = list(images_dict.keys())


# 3. Create the training data in the correct order
X = []
y = []

for rico_id in all_ids:
    if rico_id in label_dict:
 
        X.append(images_dict[rico_id])

        # Construct the array of ratings
        ratings_array = [
            label_dict[rico_id]['aesthetics_rating'],
            label_dict[rico_id]['learnability'],
            label_dict[rico_id]['efficency'],
            label_dict[rico_id]['usability_rating'],
            label_dict[rico_id]['design_quality_rating']
        ]
        y.append(ratings_array)

# Convert the lists to numpy arrays of floats
X = np.array(X, dtype=np.float32)
y = np.array(y, dtype=np.float32)

print(f"Created arrays for {len(X)} images matching 'rico_id' and the 5 rating columns.")

# convert NaN to 0 (if any)
X = np.nan_to_num(X)
y = np.nan_to_num(y)


# 4. Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(
    X, 
    y, 
    test_size=0.2, 
    random_state=42,
    shuffle=True
)

print("Split the data into training and validation sets.")
print(f"Training set: {len(X_train)} images")
print(f"Validation set: {len(X_val)} images")


# 5. Define the model
def create_simple_cnn(input_shape=(224, 224, 3), output_dim=5):
    """
    output_dim=5, z.B. [ aesth, learn, effic, usability, design_quality ]
    """
    model = models.Sequential()
    # Define an explicit input layer
    model.add(tf.keras.Input(shape=input_shape))

    # First convolution layer
    model.add(layers.Conv2D(32, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))

    # Second convolution layer
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))

    # flatten and dense-Layer
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(output_dim))  

    return model

model = create_simple_cnn()


# 6. Compile and train the model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), 
             loss='mse', 
             metrics=['mae'])

print("Starting training...")
start_time = time.time()
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=5,
    batch_size=16
)
end_time = time.time()
print(f"Training completed in {end_time - start_time:.2f} seconds.")


# 7. Save the model
model_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'model'))
os.makedirs(model_dir, exist_ok=True)

model_path = os.path.join(model_dir, 'my_model.keras')
model.save(model_path)

print(f"Model saved to {model_path}.")

# 8. Evaluate the model
def evaluate_model(model_path=model_path):
    # Load the model
    model = tf.keras.models.load_model(model_path)
    

    # make the predictions
    predictions = model.predict(X_val)
    
    # calculate the error metrics
    mse = mean_squared_error(y_val, predictions)
    mae = mean_absolute_error(y_val, predictions)
    
    print(f"Test MSE: {mse:.4f}")
    print(f"Test MAE: {mae:.4f}")

evaluate_model()