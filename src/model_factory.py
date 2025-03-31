from keras import models, layers, Model
import tensorflow as tf


import numpy as np

aesthetics_mean = 0.574732
learnability_mean = 0.601879
efficiency_mean = 0.603826
usability_mean = 0.581544
design_quality_mean = 0.581577

N_edge_features = 50176

# put them in a float32 array:
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


def create_mean_baseline_model(input_shape, means=output_means, trainable=False):
    """
    build a mean keras model, always returning the mean of the dimension (hardcoded).

    Args:
        means (list or np.array): the mean for every dimension in a 1D array.
        trainable (bool): Usually false.
    """
    # define a dummy input layer. The shape can be anything as it will be ignored.
    inputs = tf.keras.Input(shape=input_shape)
    outputs = MeanBaselineLayer(means, trainable=trainable)(inputs)
    model = tf.keras.Model(inputs, outputs, name="mean_baseline_model")
    return model, "mean-baseline"


def create_simple_cnn(input_shape=(224, 224, 3), output_dim=5):
    model = models.Sequential()
    # define an explicit input layer
    model.add(tf.keras.Input(shape=input_shape))

    # first convolution layer
    model.add(layers.Conv2D(32, (3, 3), activation="relu"))
    model.add(layers.MaxPooling2D((2, 2)))

    # second convolution layer
    model.add(layers.Conv2D(64, (3, 3), activation="relu"))
    model.add(layers.MaxPooling2D((2, 2)))

    # flatten and dense-Layer
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation="relu"))
    model.add(layers.Dense(output_dim))

    return model, "simple-cnn"


# As the dataset probably does not support a CNN using simple pixel inputs, we now want to use a pre-trained model.
def create_pretrained_resnet_cnn(
    input_shape=(224, 224, 3), output_dim=5, trainable=False
):
    """
    output_dim=5, z.B. [ aesth, learn, effic, usability, design_quality ]
    """
    resnet = tf.keras.applications.ResNet152V2(
        include_top=False, weights="imagenet", input_shape=input_shape
    )

    resnet.trainable = trainable

    # build the full model
    model = models.Sequential(
        [
            resnet,
            layers.GlobalAveragePooling2D(),
            layers.Dense(512, activation="relu"),
            layers.Dropout(0.5),
            layers.BatchNormalization(),
            layers.Dense(256, activation="relu"),
            layers.Dropout(0.5),
            layers.Dense(output_dim, activation="linear"),
        ]
    )

    return model, "resnet-cnn"


def create_pretrained_resnet_cnn_with_features(
    input_shape=(224, 224, 3),
    edge_input_shape=(N_edge_features,),
    histogram_input_shape=(512,),
    output_dim=5,
    trainable=False,
):
    """
    Creates a combined model on top of a resnet base model. This gets histogram and edge data
    as additional inputs.

     Args:
         input_shape (tuple): Form der Bilddaten.
         output_dim (int): Anzahl der Ausgabedimensionen.
         trainable (bool): Ob die ResNet-Gewichte trainierbar sind.
         histogram_input_shape (tuple): Form der Histogramm-Daten.
         edge_input_shape (tuple): shape of edge data.

     Returns:
         model (tf.keras.Model): combined model.
         str: base name of the model.
    """

    # image input and resnet
    image_input = layers.Input(shape=input_shape, name="image_input")
    resnet = tf.keras.applications.ResNet152V2(
        include_top=False, weights="imagenet", input_shape=input_shape
    )
    resnet.trainable = trainable
    x = resnet(image_input)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(512, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.5)(x)

    # histogram input
    histogram_input = layers.Input(shape=histogram_input_shape, name="histogram_input")
    h = layers.Dense(128, activation="relu")(histogram_input)
    h = layers.BatchNormalization()(h)
    h = layers.Dropout(0.3)(h)

    # edge input
    edge_input = layers.Input(shape=edge_input_shape, name="edge_input")
    e = layers.Flatten()(edge_input)
    e = layers.Dense(128, activation="relu")(e)
    e = layers.BatchNormalization()(e)
    e = layers.Dropout(0.3)(e)

    # combine features
    combined = layers.concatenate([x, h, e])

    # more dense layers
    combined = layers.Dense(512, activation="relu")(combined)
    combined = layers.Dropout(0.5)(combined)
    combined = layers.BatchNormalization()(combined)
    combined = layers.Dense(output_dim, activation="linear")(combined)

    # define model
    model = Model(
        inputs=[image_input, histogram_input, edge_input],
        outputs=combined,
        name="resnet_cnn_with_features",
    )

    return model, "resnet-cnn-with-features"


def create_cnn_softmax_output(num_classes, input_shape=(224, 224, 3)):
    """
    Creates a CNN model with softmax output for a classification task.
    No dropout.

    Args:
        num_classes (int): Number of classes.
        input_shape (tuple): Shape of the image data.

    Returns:
        model (tf.keras.Model): The CNN model.
        str: Base name of the model.
    """
    model = models.Sequential()
    # define an explicit input layer
    model.add(tf.keras.Input(shape=input_shape))

    # first convolutional layer
    model.add(layers.Conv2D(32, (3, 3), activation="relu"))
    model.add(layers.MaxPooling2D((2, 2)))

    # second convolutional layer
    model.add(layers.Conv2D(64, (3, 3), activation="relu"))
    model.add(layers.MaxPooling2D((2, 2)))

    # flatten and dense layer (without dropout)
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation="relu"))

    # final output layer with softmax activation for classification
    model.add(layers.Dense(num_classes, activation="softmax"))

    return model, "cnn-softmax"


def create_resnet_cnn_softmax_output(
    num_classes, input_shape=(224, 224, 3), trainable=False
):
    """
    Creates a CNN model with a pre-trained ResNet base model and softmax output for a classification task.

    Args:
        num_classes (int): Number of classes.
        input_shape (tuple): Shape of the image data.
        trainable (bool): Whether the ResNet weights should be trainable.

    Returns:
        model (tf.keras.Model): The ResNet-based CNN model.
        str: Base name of the model.
    """
    # create the ResNet base model
    resnet = tf.keras.applications.ResNet152V2(
        include_top=False, weights="imagenet", input_shape=input_shape
    )
    resnet.trainable = trainable

    # build the full model
    model = models.Sequential(
        [
            resnet,
            layers.GlobalAveragePooling2D(),
            layers.Dense(128, activation="relu"),
            layers.Dropout(0.5),
            layers.BatchNormalization(),
            layers.Dense(64, activation="relu"),
            layers.Dropout(0.3),
            layers.Dense(num_classes, activation="softmax"),
        ]
    )

    return model, "resnet-cnn-softmax"


def create_simpler_cnn_softmax_output(num_classes, input_shape=(224, 224, 3)):
    model = models.Sequential(
        [
            tf.keras.Input(shape=input_shape),
            layers.Conv2D(16, (3, 3), activation="relu", padding="same"),
            layers.MaxPooling2D((4, 4)),
            layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
            layers.MaxPooling2D((4, 4)),
            layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
            layers.MaxPooling2D((4, 4)),
            layers.Flatten(),
            layers.Dense(128, activation="relu"),
            layers.Dropout(0.5),
            layers.Dense(num_classes, activation="softmax"),
        ]
    )
    return model, "simple-cnn-softmax"
