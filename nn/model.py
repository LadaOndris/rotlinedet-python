import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow.keras import layers

from nn.data_loader import DataLoader


def get_model(image_size):
    """
    Creates a convolutional network that processes the input images
    with various filter sizes in the first layer. Then, it
    combines all results together, and processes it with
    a dense layer, resulting in a single prediction value.
    """
    input_shape = (image_size[0], image_size[1], 3)  # Input shape for variable-sized images

    # Input layer
    inputs = tf.keras.Input(shape=input_shape)

    # List of filter sizes for the first convolutional layer
    filter_sizes = [3, 5, 7]

    # List to store the outputs of the first convolutional layer
    outputs = []

    # Apply convolutional layers with different filter sizes
    for filter_size in filter_sizes:
        conv_layer = layers.Conv2D(filters=16, kernel_size=filter_size, strides=1, padding="same", activation="relu")(
            inputs)
        flatenned_output = layers.Flatten()(conv_layer)
        outputs.append(flatenned_output)

    # Combine all outputs
    combined = layers.concatenate(outputs)

    # Dense layer for final prediction
    dense = layers.Dense(1, activation="sigmoid")(combined)

    # Create the model
    model = tf.keras.Model(inputs=inputs, outputs=dense)

    return model


if __name__ == "__main__":
    stripe_width = 45
    image_height = 480
    data_loader = DataLoader('data/without_laser/train',
                             stripe_width=stripe_width,
                             line_width_range=(1, 12),
                             length_range=(0.33, 1),
                             intensity_range=(16, 128),
                             target_image_height=image_height)
    dataset = data_loader.build_pipeline(batch_size=1)

    model = get_model(image_size=(image_height, stripe_width))
    model.summary(line_length=200)

    # Load and preprocess an image from the dataset
    for images, labels in dataset:
        image = images[0].numpy().astype(int)
        image = np.expand_dims(image, axis=0)
        image = image / 255.0  # Normalize the image pixel values
        ground_truth_label = labels[0].numpy()
        break

    predictions = model.predict(image)
    score = predictions[0]

    class_labels = ["No Line", "Line"]
    plt.subplot(1, 1, 1)
    plt.imshow(image[0])
    plt.title("Ground Truth: {}\n Prediction: {}"
              .format(class_labels[ground_truth_label], np.squeeze(score)))
    plt.axis("off")
    plt.tight_layout()
    plt.show()
