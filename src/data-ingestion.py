import os
import time
import numpy as np
import pandas as pd
import cv2

import matplotlib.pyplot as plt


def time_log(task_description, start_time) -> float:
    """Logs the duration of a task, given a start time."""
    stop_time = time.time()
    print(f"{task_description} in {stop_time - start_time:.2f} seconds.")
    return stop_time


def load_uicrit_dataset(uicrit_path) -> pd.DataFrame:
    """Loads the UICrit dataset, removes unnecessary columns and NaN values."""
    print(f"Loading uicrit dataset from {uicrit_path}")
    start_time = time.time()
    try:
        uicrit = pd.read_csv(uicrit_path)
    except FileNotFoundError:
        print(f"Error: Could not find the uicrit dataset at {uicrit_path}.")
        exit(1)
    time_log("Loaded the uicrit dataset", start_time)
    print(f"Loaded uicrit dataset with {len(uicrit)} entries.")

    # drop unnecessary columns and any nans
    uicrit = uicrit.drop(columns=["task", "comments", "comments_source"])
    print("Cut off the columns 'task', 'comments' and 'comments_source'.")
    uicrit = uicrit.dropna()
    return uicrit


def normalize_and_standardize(uicrit, rating_scales) -> pd.DataFrame:
    """Normalizes and standardizes the UICrit dataset."""
    print("Normalizing the rating columns.")
    start_time = time.time()
    for col, scale in rating_scales.items():
        if col in uicrit.columns:
            uicrit[col] = uicrit[col] / scale
    time_log("Normalized all rating columns", start_time)

    print("Standardizing the rating columns.")
    start_time = time.time()
    for col in rating_scales.keys():
        uicrit[col] = (uicrit[col] - uicrit[col].mean()) / uicrit[col].std()
    time_log("Standardized all rating columns", start_time)

    print("Variance of the ratings:")
    for col in rating_scales.keys():
        print(f"  {col}: {uicrit[col].var()}")
    return uicrit


def load_images(image_names, combined_folder) -> dict:
    """Loads the images into a dictionary."""
    print(f"Loading {len(image_names)} images from the combined folder.")
    start_time = time.time()
    images = {}
    all_loaded = True
    for image_name in image_names:
        image_path = os.path.join(combined_folder, f"{image_name}.jpg")
        image = cv2.imread(image_path)
        if image is not None:
            images[image_name] = image
        else:
            all_loaded = False
    time_log("Loaded the images", start_time)
    print(f"Loaded {len(images)} images.")
    print(f"All images loaded successfully: {all_loaded}")
    return images


def preprocess_images(images, normalize=True) -> dict:
    """Adds a black border to the images to make them square. Resizes them to 224x224 and normalizes their pixel values."""

    # adding black edges to the images to make them square
    for k, img in images.items():
        h, w, _ = img.shape
        if h < w:
            diff = w - h
            top = diff // 2
            bottom = diff - top
            images[k] = cv2.copyMakeBorder(
                img, top, bottom, 0, 0, cv2.BORDER_CONSTANT, value=[0, 0, 0]
            )
        elif w < h:
            diff = h - w
            left = diff // 2
            right = diff - left
            images[k] = cv2.copyMakeBorder(
                img, 0, 0, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0]
            )

    print("Resizing images to 896x896.")
    start_time = time.time()
    resized_images = {k: cv2.resize(img, (896, 896)) for k, img in images.items()}
    time_log("Resized all images", start_time)

    # normalizing depending on the boolean flag
    # yes, I know boolean flags are bad :-)
    if normalize:
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])

        print("Normalizing images (Zero-Mean, Unit-Variance).")
        normalized_images = {
            k: ((img.astype(np.float32) / 255.0 - mean) / std)
            for k, img in resized_images.items()
        }
        print("Standardized all images.")
        return normalized_images

    return resized_images


def perform_edge_detection(images) -> dict:
    """Performs canny edge detection on a dictionary of images."""
    print("Performing edge detection.")
    start_time = time.time()
    edges = {}
    for name, img in images.items():
        # maybe necessary to convert to uint8
        if img.dtype != np.uint8:
            img_cv = (img * 255).astype(np.uint8)
        else:
            img_cv = img
        edge_map = cv2.Canny(img_cv, 100, 200)
        edges[name] = cv2.resize(edge_map.astype(np.float32) / 255.0, (56, 56))
    time_log("Performed edge detection", start_time)
    return edges


def calculate_color_histograms(images) -> dict:
    """Calculates color histograms for a dictionary of images."""
    print("Calculating color histograms.")
    start_time = time.time()
    color_histograms = {}
    for name, img in images.items():
        img_cv = (img * 255).astype(np.uint8) if img.dtype != np.uint8 else img
        hist = cv2.calcHist(
            [img_cv], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256]
        )
        hist = cv2.normalize(hist, hist).flatten()
        color_histograms[name] = hist
    time_log("Calculated color histograms", start_time)
    return color_histograms


def save_data(
    data_dir,
    images=None,
    edge_images=None,
    color_histograms=None,
    uicrit=None,
    rating_scales=None,
) -> None:
    """Saves images, UICrit, histograms, edges and rating scales."""
    os.makedirs(data_dir, exist_ok=True)

    print(f"Saving images to {os.path.join(data_dir, 'images.npy')}.")
    try:
        if images is not None:
            print("Saving images…")
            np.save(os.path.join(data_dir, "images.npy"), images)
        if edge_images is not None:
            print("Saving edge_images…")
            np.save(os.path.join(data_dir, "edge_images.npy"), edge_images)
        if color_histograms is not None:
            print("Saving color_histograms…")
            np.save(os.path.join(data_dir, "color_histograms.npy"), color_histograms)
    except FileNotFoundError:
        print(
            f"Error: Could not save images to {os.path.join(data_dir, 'images.npy')}."
        )
        exit(1)
    print("Saved images.")

    print(f"Saving uicrit dataset to {os.path.join(data_dir, 'uicrit.csv')}.")
    try:
        if uicrit is not None:
            print("Saving UICrit dataset…")
            uicrit.to_csv(os.path.join(data_dir, "uicrit.csv"))
        if rating_scales is not None:
            print("Saving rating_scales…")
            np.save(os.path.join(data_dir, "rating_scales.npy"), rating_scales)
    except FileNotFoundError:
        print(
            f"Error: Could not save the uicrit dataset to {os.path.join(data_dir, 'uicrit.csv')}."
        )
        exit(1)
    print("Saved uicrit dataset.")


def main():
    # 1. Load the UICrit dataset
    uicrit_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "uicrit/uicrit_public.csv")
    )
    uicrit = load_uicrit_dataset(uicrit_path)

    # define the rarting scales
    rating_scales = {
        "aesthetics_rating": 10,
        "learnability": 5,
        "efficency": 5,
        "usability_rating": 10,
        "design_quality_rating": 9,
    }
    # normalize and standardize the dataset
    # uicrit = normalize_and_standardize(uicrit, rating_scales)

    # 2. load the corresponding images
    image_names = uicrit["rico_id"].values
    combined_folder = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "combined")
    )
    images = load_images(image_names, combined_folder)

    # 3. Preprocess the images
    images = preprocess_images(images, normalize=False)
    # show one image randomly
    # import random
    # import matplotlib.pyplot as plt

    # random_image = random.choice(list(images.values()))
    # plt.imshow(random_image)
    # plt.show()

    # 4. Edge Detection, Color-Histograms
    print("Skipping edge detection and histograms, for now…")
    # edge_images = perform_edge_detection(images)
    # color_histograms = calculate_color_histograms(images)

    # saving
    data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))

    # generate images for the distribution of all columns in rating scales in uicrit
    for col in rating_scales.keys():
        if col in uicrit.columns:
            uicrit[col].hist()
            plt.title(col)
            # plt.show()
            plt.savefig(os.path.join(data_dir, f"{col}.png"))
            plt.close()

    save_data(data_dir, images=images, uicrit=uicrit, rating_scales=rating_scales)


if __name__ == "__main__":
    main()
