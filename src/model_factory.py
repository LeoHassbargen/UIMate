from keras import models, layers
import tensorflow as tf


def create_simple_cnn(input_shape=(224, 224, 3), output_dim=5):
    """
    output_dim=5, z.B. [ aesth, learn, effic, usability, design_quality ]
    """
    model = models.Sequential()
    # Define an explicit input layer
    model.add(tf.keras.Input(shape=input_shape))

    # First convolution layer
    model.add(layers.Conv2D(32, (3, 3), activation="relu"))
    model.add(layers.MaxPooling2D((2, 2)))

    # Second convolution layer
    model.add(layers.Conv2D(64, (3, 3), activation="relu"))
    model.add(layers.MaxPooling2D((2, 2)))

    # flatten and dense-Layer
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation="relu"))
    model.add(layers.Dense(output_dim))

    return model, "simple-cnn"


def create_resnet_50_model(input_shape=(224, 224, 3), output_dim=5, train_base=False):
    """
    Loads a pretrained ResNet50 as a feature extractor, then adds
    Dense layers for final output_dim=5. If 'train_base' is True,
    the base ResNet layers will be unfrozen and also trained.
    """
    base_model = tf.keras.applications.resnet50.ResNet50(
        include_top=False, weights="imagenet", input_shape=input_shape
    )
    # Freeze all base layers unless training the entire ResNet
    base_model.trainable = train_base

    model = models.Sequential()
    model.add(base_model)
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation="relu"))
    model.add(layers.Dense(output_dim))  # e.g. 5 ratings
    return model, "resnet50-pretrained"


def create_resnet_151v2_model(
    input_shape=(224, 224, 3), output_dim=5, train_base=False
):
    """
    Loads a pretrained ResNet50 as a feature extractor, then adds
    Dense layers for final output_dim=5. If 'train_base' is True,
    the base ResNet layers will be unfrozen and also trained.
    """
    base_model = tf.keras.applications.ResNet152V2(
        include_top=False, weights="imagenet", input_shape=input_shape
    )
    # Freeze all base layers unless training the entire ResNet
    base_model.trainable = train_base

    model = models.Sequential()
    model.add(base_model)
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation="relu"))
    model.add(layers.Dense(output_dim))  # e.g. 5 ratings
    return model, "resnet151V2-pretrained"
