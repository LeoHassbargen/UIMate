# Ingest all the data necessary to train the model. This file contains everything for the pipeline up to training the model.
# 1. Load the uicrit dataset from the csv file and cutoff the comments as they are not needed
# 2. Load the needed and available screenshots from the combined folder
# 3. Preprocess the images and transform them into a smaller format
# 4. Save the images and the uicrit dataset to the file system

# 1.
import json
import os
import pandas as pd
import time

uicrit_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "uicrit/uicrit_public.csv")
)

print(f"Loading uicrit dataset from {uicrit_path}")
start_time = time.time()
try:
    uicrit = pd.read_csv(uicrit_path)
except FileNotFoundError:
    print(f"Error: Could not find the uicrit dataset at {uicrit_path}.")
    exit(1)
stop_time = time.time()
print(f"Loaded the uicrit dataset in {stop_time - start_time} seconds.")
print(f"Loaded uicrit dataset with {len(uicrit)} entries.")

# Cutoff the columns that are not needed: comments, comments_source
uicrit = uicrit.drop(columns=["task", "comments", "comments_source"])
print(f"Cut off the columns 'task', 'comments' and 'comments_source'.")

# Remove all nan values
uicrit = uicrit.dropna()

# Define the rating scales
rating_scales = {
    "aesthetics_rating": 10,  # Scale 1-10
    "learnability": 5,  # Scale 1-5
    "efficency": 5,  # Scale 1-5
    "usability_rating": 10,  # Scale 1-10
    "design_quality_rating": 10,  # Scale 1-10
}

# Normalize the relevant rating columns
# We divide each rating by its maximum value (specified above).
# To scale the ratings to the range [0, 1].
print("Normalizing the rating columns.")
start_time = time.time()
for col, scale in rating_scales.items():
    if col in uicrit.columns:
        uicrit[col] = uicrit[col] / scale
        print(f"Normalized '{col}' by a factor of {scale}.")

stop_time = time.time()
print(f"Normalized all rating columns in {stop_time - start_time} seconds.")

# Standardize the ratings
# We subtract the mean and divide by the standard deviation.
# This ensures that the ratings have a mean of 0 and a standard deviation of 1.

print("Standardizing the rating columns.")
start_time = time.time()

for col in rating_scales.keys():
    uicrit[col] = (uicrit[col] - uicrit[col].mean()) / uicrit[col].std()

stop_time = time.time()
print(f"Standardized all rating columns in {stop_time - start_time} seconds.")

# Print the variance for each dimension
print(f"Variance of the ratings:")
for col in rating_scales.keys():
    print(f"  {col}: {uicrit[col].var()}")

# 2.
import cv2
import numpy as np

image_names = uicrit["rico_id"].values

print(f"Loading {len(image_names)} images from the combined folder.")
start_time = time.time()
all_loaded = True

# Load the images from the combined folder
images = {}
start_time = time.time()
for image_name in image_names:
    image_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "combined", f"{image_name}.jpg")
    )
    image = cv2.imread(image_path)

    if image is not None:
        images[image_name] = image
    else:
        all_loaded = False


stop_time = time.time()

print(f"Loaded the images in {stop_time - start_time} seconds.")

print(f"Loaded {len(images)} images.")
print(f"All images loaded successfully: {all_loaded}")
# 3.

# Resize the images to 224x224
print("Resizing images to 224x224.")
start_time = time.time()
images = {key: cv2.resize(image, (224, 224)) for key, image in images.items()}
stop_time = time.time()
print(f"Resized all images in {stop_time - start_time} seconds.")

# Normalize the images
print("Normalizing images.")
images = {key: image / 255.0 for key, image in images.items()}
print("Normalized all images.")

# Convert the images to numpy arrays
images = {key: np.array(image) for key, image in images.items()}

# perform image transformation before edge detection
print("Performing image transformations.")

# check if the first image is of dtype np.uint8
if images[image_names[0]].dtype != np.uint8:

    # convert the images to np.uint8
    images_cv = {key: (image * 255).astype(np.uint8) for key, image in images.items()}
else:
    images_cv = images

# perform edge detection and store separately using the image_name
print("Performing edge detection.")
start_time = time.time()

edge_images = {}
for image_name, image in images_cv.items():
    edges = cv2.Canny(image, 100, 200)
    edges = edges.astype(np.float32) / 255.0
    edge_pooled = cv2.resize(edges, (56, 56))
    edge_images[image_name] = edge_pooled

stop_time = time.time()
print(f"Performed edge detection in {stop_time - start_time} seconds.")

# Convert the edge images to numpy arrays
edge_images = {key: np.array(image) for key, image in edge_images.items()}

# calculate the color histograms and store separately using the image_name
print("Calculating color histograms.")
start_time = time.time()

color_histograms = {}
for image_name, image in images_cv.items():
    hist = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    color_histograms[image_name] = hist

stop_time = time.time()
print(f"Calculated color histograms in {stop_time - start_time} seconds.")

# Ensure the data directory exists
data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
os.makedirs(data_dir, exist_ok=True)

# Save both the images and the uicrit dataset to the file system
print(f"Saving images to {os.path.join(data_dir, 'images.npy')}.")
try:
    np.save(os.path.join(data_dir, "images.npy"), images)
    np.save(os.path.join(data_dir, "edge_images.npy"), edge_images)
    np.save(os.path.join(data_dir, "color_histograms.npy"), color_histograms)
except FileNotFoundError:
    print(
        f"Error: Could not save the images to {os.path.join(data_dir, 'images.npy')}."
    )
    exit(1)
print("Saved images.")

print(f"Saving uicrit dataset to {os.path.join(data_dir, 'uicrit.csv')}.")
try:
    uicrit.to_csv(os.path.join(data_dir, "uicrit.csv"))
    np.save(os.path.join(data_dir, "rating_scales.npy"), rating_scales)
except FileNotFoundError:
    print(
        f"Error: Could not save the uicrit dataset to {os.path.join(data_dir, 'uicrit.csv')}."
    )
    exit(1)
print("Saved uicrit dataset.")
