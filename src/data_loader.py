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

    images_histograms = np.load(
        os.path.abspath(
            os.path.join(
                os.path.dirname(__file__), "..", "data", "color_histograms.npy"
            )
        ),
        allow_pickle=True,
    ).item()

    images_edges = np.load(
        os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "data", "edge_images.npy")
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

common_ids = (
    set(images_dict.keys())
    & set(images_histograms.keys())
    & set(images_edges.keys())
    & set(uicrit["rico_id"])
)

print(f"Anzahl der gemeinsamen 'rico_id's: {len(common_ids)}")

# initialize empty lists
X_images = []
X_histograms = []
X_edges = []
y = []

# iterate over rico_ids
for rico_id in common_ids:
    # images
    image = images_dict[rico_id].astype(np.float32) / 255.0
    X_images.append(image)

    # histograms
    hist = images_histograms[rico_id].astype(np.float32)
    X_histograms.append(hist)

    # edges
    edges = images_edges[rico_id].astype(np.float32)
    X_edges.append(edges)

    # ratings
    ratings_array = [
        uicrit.loc[uicrit["rico_id"] == rico_id, "aesthetics_rating"].values[0],
        uicrit.loc[uicrit["rico_id"] == rico_id, "learnability"].values[0],
        uicrit.loc[uicrit["rico_id"] == rico_id, "efficency"].values[0],
        uicrit.loc[uicrit["rico_id"] == rico_id, "usability_rating"].values[0],
        uicrit.loc[uicrit["rico_id"] == rico_id, "design_quality_rating"].values[0],
    ]
    y.append(ratings_array)

# convert lists to numpy arrays
X_images = np.array(X_images, dtype=np.float32)
X_histograms = np.array(X_histograms, dtype=np.float32)
X_edges = np.array(X_edges, dtype=np.float32)
y = np.array(y, dtype=np.float32)

print(
    f"Created arrays for {len(X_images)} images matching 'rico_id' and the 5 rating columns."
)

# check missing values
if np.isnan(X_histograms).any() or np.isnan(X_edges).any() or np.isnan(y).any():
    print(
        "Warnung: Fehlende Werte in den Daten erkannt. Bitte bereinigen Sie die Daten."
    )
    mask = ~(
        np.isnan(X_histograms).any(axis=1)
        | np.isnan(X_edges).any(axis=1)
        | np.isnan(y).any(axis=1)
    )
    X_images = X_images[mask]
    X_histograms = X_histograms[mask]
    X_edges = X_edges[mask]
    y = y[mask]
    print(f"Anzahl der Beispiele nach Entfernen fehlender Werte: {len(X_images)}")
else:
    print("No missing values.")

# split
(
    X_train_images,
    X_val_images,
    X_train_histograms,
    X_val_histograms,
    X_train_edges,
    X_val_edges,
    y_train,
    y_val,
) = train_test_split(
    X_images,
    X_histograms,
    X_edges,
    y,
    test_size=VAL_SIZE,
    random_state=42,
    shuffle=True,
)

# structure as dicts
X_train = {
    "image_input": X_train_images,
    "histogram_input": X_train_histograms,
    "edge_input": X_train_edges,
}

X_val = {
    "image_input": X_val_images,
    "histogram_input": X_val_histograms,
    "edge_input": X_val_edges,
}


def get_training_data():
    return X_train, y_train


def get_validation_data():
    return X_val, y_val


def get_evaluation_split_size():
    return VAL_SIZE


def get_mean_ratings():
    return uicrit.mean()[-5:].values
