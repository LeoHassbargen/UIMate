from keras import models, layers
import tensorflow as tf


import numpy as np
import tensorflow as tf

aesthetics_mean = 0.574732
learnability_mean = 0.601879
efficiency_mean = 0.603826
usability_mean = 0.581544
design_quality_mean = 0.581577

# Put them in a float32 array:
output_means = np.array(
    [
        aesthetics_mean,
        learnability_mean,
        efficiency_mean,
        usability_mean,
        design_quality_mean,
    ],
    dtype=np.float32,
)


class MeanBaselineLayer(tf.keras.layers.Layer):
    """
    A custom Keras layer that always outputs a fixed vector of means,
    regardless of the input.
    """

    def __init__(self, means, trainable=False, **kwargs):
        super().__init__(**kwargs)

        # Store the means in a TF variable. By default, it's non-trainable
        # for a "true" baseline. If you set trainable=True, the model can
        # learn the best constant offset during fit.
        self.means = tf.Variable(
            initial_value=means,
            trainable=trainable,
            dtype=tf.float32,
            name="fixed_means",
        )

    def call(self, inputs):
        # inputs.shape -> (batch_size, anything)
        batch_size = tf.shape(inputs)[0]
        # Repeat the means for each sample in the batch
        return tf.tile(tf.reshape(self.means, [1, -1]), [batch_size, 1])

    def get_config(self):
        """
        Return a dictionary containing the configuration used to initialize this layer.
        """
        base_config = super().get_config()
        # Convert the tensor to a Python list so it can be serialized in JSON.
        means_list = self.means.numpy().tolist()

        config = {**base_config, "means": means_list, "trainable": self.means.trainable}
        return config

    @classmethod
    def from_config(cls, config):
        """
        Recreate the layer from its config. We pop 'means' and 'trainable'
        from config, then pass them in to the constructor.
        """
        means = config.pop("means")
        trainable = config.pop("trainable", False)
        # Convert means back to tf-compatible type
        means_tf = tf.constant(means, dtype=tf.float32)
        return cls(means=means_tf, trainable=trainable, **config)


def build_mean_baseline_model(input_shape, means=output_means, trainable=False):
    """
    Builds a Keras model whose output is always `means`.

    Args:
        means (list or np.array): 1D array of length = number of target dims
        trainable (bool): Whether we allow this constant to be adjusted
                          during `model.fit()`. Usually False for a pure
                          baseline, True if you want to let it "learn"
                          the best constant offset.
    """
    # Define a dummy input layer. The shape can be anything as it will be ignored.
    inputs = tf.keras.Input(shape=input_shape)
    outputs = MeanBaselineLayer(means, trainable=trainable)(inputs)
    model = tf.keras.Model(inputs, outputs, name="mean_baseline_model")
    return model, "mean-baseline"


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
