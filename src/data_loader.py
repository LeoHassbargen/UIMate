import os
import numpy as np
import pandas as pd

# global configuration for splitting (not used anymore) and rating_scales variable
VAL_SIZE = float(os.getenv("VAL_SIZE", 0.2))
rating_scales = None  # global rating scales, if needed


# -----------------------------------------------------------------------------
# Section 1: File Loading Helpers
# -----------------------------------------------------------------------------
def load_numpy_item(filepath: str, description: str) -> dict:
    filepath = os.path.abspath(filepath)
    try:
        data = np.load(filepath, allow_pickle=True).item()
    except FileNotFoundError:
        import logging

        logging.error(f"Could not find {description} at {filepath}.")
        exit(1)
    print(f"Loaded {description} from {filepath}.")
    return data


def load_csv_data(filepath: str, description: str) -> pd.DataFrame:
    filepath = os.path.abspath(filepath)
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        import logging

        logging.error(f"Could not find {description} at {filepath}.")
        exit(1)
    print(f"Loaded {description} from {filepath}.")
    return df


# -----------------------------------------------------------------------------
# Section 2: Data Preparation Helpers
# -----------------------------------------------------------------------------
def get_common_ids(
    images_dict: dict,
    images_histograms: dict,
    images_edges: dict,
    uicrit: pd.DataFrame,
) -> set:
    image_ids = set(images_dict.keys())
    histogram_ids = set(images_histograms.keys())
    edge_ids = set(images_edges.keys())
    uicrit_ids = set(uicrit["rico_id"])

    all_ids = image_ids & histogram_ids & edge_ids & uicrit_ids

    missing_from_images = histogram_ids - image_ids
    missing_from_histograms = image_ids - histogram_ids
    missing_from_edges = image_ids - edge_ids
    missing_from_uicrit = image_ids - uicrit_ids

    if (
        missing_from_images
        or missing_from_histograms
        or missing_from_edges
        or missing_from_uicrit
    ):
        import logging

        logging.warning("Missing rico_ids detected!")
        if missing_from_images:
            logging.warning(f"Missing from images: {missing_from_images}")
        if missing_from_histograms:
            logging.warning(f"Missing from histograms: {missing_from_histograms}")
        if missing_from_edges:
            logging.warning(f"Missing from edges: {missing_from_edges}")
        if missing_from_uicrit:
            logging.warning(f"Missing from uicrit: {missing_from_uicrit}")

    common = all_ids
    print(f"Number of common 'rico_id's: {len(common)}")
    return common


def prepare_arrays(
    common_ids: set,
    images_dict: dict,
    images_histograms: dict,
    images_edges: dict,
    uicrit: pd.DataFrame,
) -> tuple:
    X_images, X_histograms, X_edges, y = [], [], [], []
    for _, row in uicrit.iterrows():
        rico_id = row["rico_id"]
        X_images.append(images_dict[rico_id].astype(np.float32))
        X_histograms.append(images_histograms[rico_id].astype(np.float32))
        X_edges.append(images_edges[rico_id].astype(np.float32))
        ratings = [
            row["aesthetics_rating"],
            row["learnability"],
            row["efficency"],
            row["usability_rating"],
            row["design_quality_rating"],
        ]
        y.append(ratings)
    # Convert to numpy arrays
    X_images = np.array(X_images, dtype=np.float32)
    X_histograms = np.array(X_histograms, dtype=np.float32)
    X_edges = np.array(X_edges, dtype=np.float32)
    y = np.array(y, dtype=np.int32) - 1  # convert classes to 0-based indexing
    print(
        f"Created arrays for {len(X_images)} images matching 'rico_id' and 5 rating columns, with {len(y)} samples."
    )
    return X_images, X_histograms, X_edges, y


def check_and_remove_missing_values(
    X_images: np.ndarray,
    X_histograms: np.ndarray,
    X_edges: np.ndarray,
    y: np.ndarray,
) -> tuple:
    if np.isnan(X_histograms).any() or np.isnan(X_edges).any() or np.isnan(y).any():
        print("Warning: Missing values detected. Removing affected examples.")
        mask = ~(
            np.isnan(X_histograms).any(axis=1)
            | np.isnan(X_edges).any(axis=1)
            | np.isnan(y).any(axis=1)
        )
        X_images = X_images[mask]

        X_edges = X_edges[mask]
        y = y[mask]
        print(f"Examples remaining: {len(X_images)}")
        if len(X_images) == 0:
            raise ValueError("No examples remaining after removing missing values.")
    else:
        print("No missing values detected.")
    return X_images, X_histograms, X_edges, y
    return X_images, X_histograms, X_edges, y


def create_input_dictionaries(
    X_images: np.ndarray, X_histograms: np.ndarray, X_edges: np.ndarray
) -> dict:
    return {
        "image_input": X_images,
        "histogram_input": X_histograms,
        "edge_input": X_edges,
    }


def undersample_classes(data, max_samples_per_class=None, seed=42):
    """
    Undersamples the data to balance classes by taking at most
    max_samples_per_class examples from each class.

    Args:
        data: List of (rico_id, label) pairs
        max_samples_per_class: Maximum samples per class (if None, no undersampling)
        seed: Random seed for reproducibility

    Returns:
        List of undersampled (rico_id, label) pairs
    """
    if max_samples_per_class is None:
        return data

    # group data by class labels
    class_groups = {}
    for rico_id, label in data:
        if label not in class_groups:
            class_groups[label] = []
        class_groups[label].append((rico_id, label))

    # shuffle each group with a consistent seed
    random_state = np.random.RandomState(seed)
    for label in class_groups:
        random_state.shuffle(class_groups[label])

    # take at most max_samples_per_class from each group
    undersampled_data = []
    for label in sorted(class_groups.keys()):
        samples = class_groups[label][:max_samples_per_class]
        undersampled_data.extend(samples)
        print(f"Class {label}: {len(class_groups[label])} → {len(samples)} samples")

    # shuffle the final result
    random_state.shuffle(undersampled_data)

    return undersampled_data


def create_undersampled_tf_dataset(
    common_ids,
    images_dict,
    uicrit,
    dimension_index=0,
    images_histograms=None,
    images_edges=None,
    batch_size=32,
    shuffle=True,
    cache=False,
    validation_split=0.2,
    seed=42,
    max_samples_per_class=None,
):
    """
    Creates a TensorFlow Dataset with train/validation split for a specific dimension
    with undersampling applied to the training set only.

    Args:
        common_ids: Set of common rico_ids
        images_dict, images_histograms, images_edges: Input data dictionaries
        uicrit: Pandas DataFrame with ratings
        dimension_index: Which rating dimension to use (0=aesthetics, 1=learnability, etc.)
        batch_size: Batch size for training
        shuffle: Whether to shuffle the data
        cache: Whether to cache the dataset
        validation_split: Proportion of data to use for validation
        seed: Random seed for reproducibility
        max_samples_per_class: Maximum samples per class for undersampling (None = no undersampling)

    Returns:
        A tuple of (train_dataset, validation_dataset, train_samples, val_samples)
    """
    # first, collect all rico_ids and ratings in common
    filtered_data = []
    dimension_names = [
        "aesthetics_rating",
        "learnability",
        "efficency",
        "usability_rating",
        "design_quality_rating",
    ]
    dim_name = dimension_names[dimension_index]

    for _, row in uicrit.iterrows():
        rico_id = row["rico_id"]
        if rico_id in common_ids:
            ratings = [
                row["aesthetics_rating"],
                row["learnability"],
                row["efficency"],
                row["usability_rating"],
                row["design_quality_rating"],
            ]
            # get specified dimension and convert to 0-based
            label = ratings[dimension_index] - 1
            filtered_data.append((rico_id, label))

    # shuffle the filtered data
    import random

    random.seed(seed)
    random.shuffle(filtered_data)

    # split into train and validation BEFORE undersampling
    # this ensures validation data represents the true distribution
    split_idx = int(len(filtered_data) * (1 - validation_split))
    train_data = filtered_data[:split_idx]
    val_data = filtered_data[split_idx:]

    print(
        f"Original split: {len(train_data)} training samples, {len(val_data)} validation samples"
    )

    # apply undersampling to training data only
    if max_samples_per_class is not None:
        print(
            f"Undersampling dimension '{dim_name}' with max {max_samples_per_class} samples per class..."
        )
        train_data = undersample_classes(
            train_data, max_samples_per_class=max_samples_per_class, seed=seed
        )
        print(
            f"After undersampling: {len(train_data)} training samples (validation unchanged)"
        )

    # create training generator
    def train_generator():
        for rico_id, label in train_data:
            # Get image for this rico_id
            image = images_dict[rico_id].astype(np.float32)
            yield image, label

    # create validation generator
    def val_generator():
        for rico_id, label in val_data:
            image = images_dict[rico_id].astype(np.float32)
            yield image, label

    # get output shape and number of classes
    sample_image = images_dict[next(iter(common_ids))].astype(np.float32)
    image_shape = sample_image.shape

    # create train dataset
    train_ds = tf.data.Dataset.from_generator(
        train_generator,
        output_signature=(
            tf.TensorSpec(shape=image_shape, dtype=tf.float32),
            tf.TensorSpec(shape=(), dtype=tf.int32),
        ),
    )

    # create validation dataset
    val_ds = tf.data.Dataset.from_generator(
        val_generator,
        output_signature=(
            tf.TensorSpec(shape=image_shape, dtype=tf.float32),
            tf.TensorSpec(shape=(), dtype=tf.int32),
        ),
    )

    # apply optimizations
    if cache:
        train_ds = train_ds.cache()
        val_ds = val_ds.cache()

    if shuffle:
        train_ds = train_ds.shuffle(buffer_size=min(1000, len(train_data)))

    # one-hot encode the labels for categorical_crossentropy
    num_classes = len(set(label for _, label in filtered_data))
    train_ds = train_ds.map(lambda x, y: (x, tf.one_hot(y, num_classes)))
    val_ds = val_ds.map(lambda x, y: (x, tf.one_hot(y, num_classes)))

    train_samples = len(train_data)
    val_samples = len(val_data)

    print(
        f"Final dataset: {train_samples} training samples, {val_samples} validation samples"
    )
    print(f"Number of classes: {num_classes}")

    # batch, repeat, and prefetch
    train_ds = train_ds.batch(batch_size).repeat().prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return train_ds, val_ds, train_samples, val_samples


# -----------------------------------------------------------------------------
# Section 4: Exposed Data Access Functions
# -----------------------------------------------------------------------------
def get_mean_ratings(uicrit_df: pd.DataFrame) -> np.ndarray:
    return uicrit_df.mean()[-5:].values


import tensorflow as tf


def data_generator(common_ids, images_dict, images_histograms, images_edges, uicrit):
    """
    Generator that yields a tuple (input_dict, label) for each sample.
    Only samples with a common rico_id are yielded.
    """
    for _, row in uicrit.iterrows():
        rico_id = row["rico_id"]
        if rico_id in common_ids:
            # load and cast each component on the fly
            image = images_dict[rico_id].astype(np.float32)
            histogram = images_histograms[rico_id].astype(np.float32)
            edge = images_edges[rico_id].astype(np.float32)
            ratings = [
                row["aesthetics_rating"],
                row["learnability"],
                row["efficency"],
                row["usability_rating"],
                row["design_quality_rating"],
            ]
            # convert ratings to 0-based indexing
            label = np.array(ratings, dtype=np.int32) - 1
            yield {
                "image_input": image,
                "histogram_input": histogram,
                "edge_input": edge,
            }, label


def data_generator(
    common_ids,
    images_dict,
    uicrit,
    dimension_index=None,
    images_histograms=None,
    images_edges=None,
):
    """
    Generator that yields a tuple (input_dict, label) for each sample.
    If dimension_index is provided, only returns the label for that specific dimension.
    """
    for _, row in uicrit.iterrows():
        rico_id = row["rico_id"]
        if rico_id in common_ids:
            # load and cast each component on the fly
            image = images_dict[rico_id].astype(np.float32)
            if images_histograms:
                histogram = images_histograms[rico_id].astype(np.float32)
            if images_edges:
                edge = images_edges[rico_id].astype(np.float32)
            ratings = [
                row["aesthetics_rating"],
                row["learnability"],
                row["efficency"],
                row["usability_rating"],
                row["design_quality_rating"],
            ]
            # convert ratings to 0-based indexing
            label = np.array(ratings, dtype=np.int32) - 1

            # option to return only one dimension's label
            if dimension_index is not None:
                label = label[dimension_index]

            input_dict = {"image_input": image}
            if images_histograms is not None:
                input_dict["histogram_input"] = histogram
            if images_edges is not None:
                input_dict["edge_input"] = edge
            yield input_dict, label


def create_tf_dataset(
    common_ids,
    images_dict,
    uicrit,
    dimension_index=None,
    images_histograms=None,
    images_edges=None,
    batch_size=32,
    shuffle=True,
    cache=False,
    validation_split=0.2,
    seed=42,
):
    """
    Creates a TensorFlow Dataset with train/validation split for a specific dimension.

    Args:
        common_ids: Set of common rico_ids
        images_dict, images_histograms, images_edges: Input data dictionaries
        uicrit: Pandas DataFrame with ratings
        dimension_index: Which rating dimension to use (0=aesthetics, 1=learnability, etc.)
        batch_size: Batch size for training
        shuffle: Whether to shuffle the data
        cache: Whether to cache the dataset
        validation_split: Proportion of data to use for validation
        seed: Random seed for reproducibility

    Returns:
        A tuple of (train_dataset, validation_dataset)
    """
    # first, collect all rico_ids and ratings in common
    filtered_data = []
    for _, row in uicrit.iterrows():
        rico_id = row["rico_id"]
        if rico_id in common_ids:
            ratings = [
                row["aesthetics_rating"],
                row["learnability"],
                row["efficency"],
                row["usability_rating"],
                row["design_quality_rating"],
            ]
            # get specified dimension and convert to 0-based
            label = ratings[dimension_index] - 1
            filtered_data.append((rico_id, label))

    # shuffle the filtered data
    import random

    random.seed(seed)
    random.shuffle(filtered_data)

    # split into train and validation
    split_idx = int(len(filtered_data) * (1 - validation_split))
    train_data = filtered_data[:split_idx]
    val_data = filtered_data[split_idx:]

    print(
        f"Split data: {len(train_data)} training samples, {len(val_data)} validation samples"
    )

    # create training generator that yields only image_input and the label
    def train_generator():
        for rico_id, label in train_data:
            # Get image for this rico_id
            image = images_dict[rico_id].astype(np.float32)
            # Convert label to one-hot if needed
            yield image, label

    # create validation generator
    def val_generator():
        for rico_id, label in val_data:
            image = images_dict[rico_id].astype(np.float32)
            yield image, label

    # get output shape and number of classes
    sample_image = images_dict[next(iter(common_ids))].astype(np.float32)
    image_shape = sample_image.shape

    # create train dataset
    train_ds = tf.data.Dataset.from_generator(
        train_generator,
        output_signature=(
            tf.TensorSpec(shape=image_shape, dtype=tf.float32),
            tf.TensorSpec(shape=(), dtype=tf.int32),
        ),
    )

    # create validation dataset
    val_ds = tf.data.Dataset.from_generator(
        val_generator,
        output_signature=(
            tf.TensorSpec(shape=image_shape, dtype=tf.float32),
            tf.TensorSpec(shape=(), dtype=tf.int32),
        ),
    )

    # apply optimizations
    if cache:
        train_ds = train_ds.cache()
        val_ds = val_ds.cache()

    if shuffle:
        train_ds = train_ds.shuffle(buffer_size=min(1000, len(train_data)))

    # one-hot encode the labels for categorical_crossentropy
    num_classes = len(set(label for _, label in filtered_data))
    train_ds = train_ds.map(lambda x, y: (x, tf.one_hot(y, num_classes)))
    val_ds = val_ds.map(lambda x, y: (x, tf.one_hot(y, num_classes)))

    train_samples = len(train_data)
    val_samples = len(val_data)

    print(
        f"Split data: {train_samples} training samples, {val_samples} validation samples"
    )

    # batch, repeat, and prefetch
    train_ds = train_ds.batch(batch_size).repeat().prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return train_ds, val_ds, train_samples, val_samples


# -----------------------------------------------------------------------------
# Section 5: Main Data Loading Pipeline
# -----------------------------------------------------------------------------
def load_data(dimension_name="aesthetics_rating", max_samples_per_class=None) -> tuple:
    """
    Loads and prepares data for training with optional undersampling.

    Args:
        dimension_name: The name of the dimension to use (e.g., "aesthetics_rating")
        max_samples_per_class: Maximum samples per class for undersampling.
                              If None, no undersampling is performed.

    Returns:
        train_ds: Training dataset
        val_ds: Validation dataset
        uicrit: UICrit dataframe
        rating_scales_data: Rating scales dictionary
        train_samples: Number of training samples
        val_samples: Number of validation samples
    """
    # load files
    images_dict = load_numpy_item(
        os.path.join("UIMate", "data", "images.npy"), "images"
    )
    images_histograms = load_numpy_item(
        os.path.join("UIMate", "data", "color_histograms.npy"),
        "color histograms",
    )
    images_edges = load_numpy_item(
        os.path.join("UIMate", "data", "edge_images.npy"), "edge images"
    )
    rating_scales_data = load_numpy_item(
        os.path.join("UIMate", "data", "rating_scales.npy"), "rating scales"
    )
    uicrit = load_csv_data(
        os.path.join("UIMate", "data", "uicrit.csv"), "uicrit dataset"
    )
    print(f"Loaded {len(images_dict)} images and {len(uicrit)} uicrit entries.")

    # get common IDs
    common_ids = get_common_ids(images_dict, images_histograms, images_edges, uicrit)

    # retrieve dimension index
    dimension_names = [
        "aesthetics_rating",
        "learnability",
        "efficency",
        "usability_rating",
        "design_quality_rating",
    ]
    dim_idx = dimension_names.index(dimension_name)

    # prepare the data arrays using the undersampling function
    train_ds, val_ds, train_samples, val_samples = create_undersampled_tf_dataset(
        common_ids=common_ids,
        images_dict=images_dict,
        images_histograms=None,
        images_edges=None,
        uicrit=uicrit,
        dimension_index=dim_idx,
        batch_size=32,
        shuffle=True,
        max_samples_per_class=max_samples_per_class,
    )

    print(f"Data loading pipeline completed for dimension '{dimension_name}'.")
    if max_samples_per_class:
        print(
            f"Undersampling applied with max {max_samples_per_class} samples per class."
        )

    return train_ds, val_ds, uicrit, rating_scales_data, train_samples, val_samples


if __name__ == "__main__":
    load_data()
