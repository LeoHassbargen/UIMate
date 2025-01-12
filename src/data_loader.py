# 1. Load the data

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


VAL_SIZE = float(os.getenv("VAL_SIZE", 0.2))
TRAIN_SIZE = 1 - VAL_SIZE

try:
    images_dict = np.load(
        os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "data", "images.npy")
        ),
        allow_pickle=True,
    ).item()

    uicrit = pd.read_csv(
        os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "data", "uicrit.csv")
        )
    )
except FileNotFoundError:
    print("Error: Could not find the images or uicrit dataset.")
    exit(1)

print(
    f"Loaded {len(images_dict)} images and {len(uicrit)} entries from the file system."
)

# 2. Create the dictionary for the ratings
label_dict = {
    row["rico_id"]: {
        "aesthetics_rating": row["aesthetics_rating"],
        "learnability": row["learnability"],
        "efficency": row["efficency"],
        "usability_rating": row["usability_rating"],
        "design_quality_rating": row["design_quality_rating"],
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
            label_dict[rico_id]["aesthetics_rating"],
            label_dict[rico_id]["learnability"],
            label_dict[rico_id]["efficency"],
            label_dict[rico_id]["usability_rating"],
            label_dict[rico_id]["design_quality_rating"],
        ]
        y.append(ratings_array)

# Convert the lists to numpy arrays of floats
X = np.array(X, dtype=np.float32)
y = np.array(y, dtype=np.float32)

print(
    f"Created arrays for {len(X)} images matching 'rico_id' and the 5 rating columns."
)

# convert NaN to 0 (if any)
X = np.nan_to_num(X)
y = np.nan_to_num(y)


# 4. Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=VAL_SIZE, random_state=42, shuffle=True
)

print("Split the data into training and validation sets.")
print(f"Training set: {len(X_train)} images")
print(f"Validation set: {len(X_val)} images")


def get_training_data():
    return X_train, y_train


def get_validation_data():
    return X_val, y_val


def get_evaluation_split_size():
    return VAL_SIZE
