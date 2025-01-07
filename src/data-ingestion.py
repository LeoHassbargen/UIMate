# Ingest all the data necessary to train the model. This file contains everything for the pipeline up to training the model.
# 1. Load the uicrit dataset from the csv file and cutoff the comments as they are not needed
# 2. Load the needed and available screenshots from the combined folder
# 3. Preprocess the images and transform them into a smaller format
# 4. Save the images and the uicrit dataset to the file system

# 1.
import os
import pandas as pd
import time

uicrit_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'uicrit/uicrit_public.csv'))

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
uicrit = uicrit.drop(columns=['comments', 'comments_source'])
print(f"Cut off the columns 'comments' and 'comments_source'.")

# 2.
import cv2
import numpy as np

image_names = uicrit['rico_id'].values

print(f"Loading {len(image_names)} images from the combined folder.")
start_time = time.time()
all_loaded = True

# Load the images from the combined folder
images = {}
start_time = time.time()
for image_name in image_names:
    image_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'combined', f'{image_name}.jpg'))
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

# 4.

# Ensure the data directory exists
data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))
os.makedirs(data_dir, exist_ok=True)

# Save both the images and the uicrit dataset to the file system
print(f"Saving images to {os.path.join(data_dir, 'images.npy')}.")
try:
    np.save(os.path.join(data_dir, 'images.npy'), images)
except FileNotFoundError:
    print(f"Error: Could not save the images to {os.path.join(data_dir, 'images.npy')}.")
    exit(1)
print("Saved images.")

print(f"Saving uicrit dataset to {os.path.join(data_dir, 'uicrit.csv')}.")
try:
    uicrit.to_csv(os.path.join(data_dir, 'uicrit.csv'))
except FileNotFoundError:
    print(f"Error: Could not save the uicrit dataset to {os.path.join(data_dir, 'uicrit.csv')}.")
    exit(1)
print("Saved uicrit dataset.")