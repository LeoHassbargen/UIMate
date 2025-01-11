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
