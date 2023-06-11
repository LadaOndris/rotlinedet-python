import os
from typing import Tuple

import tensorflow as tf


class DataLoader:
    """
    Creates a new tensorflow.data pipeline from raw images.
    Rotates the images randomly, crops a random vertical strip, and
    in half of the cases, generates a line with random properties (intensity, length).
    """

    def __init__(self, images_path: str, stripe_width: int,
                 line_width_range: Tuple[int, int],
                 intensity_range: Tuple[int, int],
                 length_range: Tuple[float, float]):
        """
        :param images_path: Path to a folder with images that should be used for generating the dataset.
        :param stripe_width: The width of the cropped vertical stripe from the rotated image.
        :param line_width_range: Min and max width of the generated line in number of pixels.
        :param intensity_range: Min and max intensity, in the [0, 255] range.
        :param length_range: Min and max length, in the [0, 1] range.
        """
        self.images_path = images_path
        self.stripe_width = stripe_width
        self.line_width_range = line_width_range
        self.intensity_range = intensity_range
        self.length_range = length_range

    # def build_pipeline(self):
    #     # Reads images from the corresponding folder
    #
    #     # Randomly selects one
    #
    #     # Adds padding around the images such that the rotation does not
    #     # crop part of the image
    #
    #     # Rotate it randomly
    #
    #     # Select a vertical stripe at a random place by cropping the image
    #
    #     # Generate synthetic line inside of it (varying intensity, width, length)
    #     # half of the time.
    #     # Width of the line is randomly determined as per the defined range.
    #     # Intensity is randomly generated in the defined range.
    #     # Length is randomly generated in the defined range.
    #
    #     # Create a label whether it contains the line
    #
    #     # Batch the samples
    #
    #     # Return dataset pipeline
    #     pass

    def _parse_image(self, image_path):
        image = tf.io.read_file(image_path)
        image = tf.image.decode_image(image, channels=3)
        return image

    def _pad_image(self, image):
        img_height = tf.shape(image)[0]
        img_width = tf.shape(image)[1]

        # Pad the image before rotation, so that the whole image is visible
        diagonal = tf.math.sqrt(tf.math.square(tf.cast(img_height, tf.float32)) +
                                tf.math.square(tf.cast(img_width, tf.float32)))
        diagonal = tf.cast(tf.math.ceil(diagonal), tf.int32)
        vertical_pad = diagonal - img_height
        vertical_pad_half = vertical_pad // 2
        horizontal_pad = diagonal - img_width
        horizontal_pad_half = horizontal_pad // 2

        pad_values = [[vertical_pad_half, vertical_pad - vertical_pad_half],
                      [horizontal_pad_half, horizontal_pad - horizontal_pad_half],
                      [0, 0]]
        padded = tf.pad(image, pad_values, constant_values=0)
        return padded

    def _rotate_image(self, image):
        import tensorflow_addons as tfa
        rotation_angle = tf.random.uniform(shape=[], minval=0, maxval=2 * 3.14159)
        return tfa.image.rotate(image, rotation_angle)

    def _crop_vertical_stripe(self, image):
        height = tf.shape(image)[0]
        width = tf.shape(image)[1]
        x = tf.random.uniform(shape=[], minval=0, maxval=width - self.stripe_width, dtype=tf.int32)
        cropped_image = tf.image.crop_to_bounding_box(image, 0, x, height, self.stripe_width)
        return cropped_image

    def _generate_line(self, image):
        image_height = tf.shape(image)[0]
        image_width = tf.shape(image)[1]
        intensity = tf.random.uniform(shape=[], minval=self.intensity_range[0], maxval=self.intensity_range[1])
        line_width = tf.random.uniform(shape=[], minval=self.line_width_range[0], maxval=self.line_width_range[1],
                                       dtype=tf.int32)
        length_fraction = tf.random.uniform(shape=[], minval=self.length_range[0], maxval=self.length_range[1])

        line = tf.ones(shape=[image_height, line_width, 3]) * intensity
        line_length = tf.cast(tf.cast(image_height, tf.float32) * length_fraction, dtype=tf.int32)
        line_mask = tf.pad(tf.ones(shape=[line_length, line_width]), [[0, image_height - line_length], [0, 0]])
        line_mask = tf.expand_dims(line_mask, axis=-1)
        line_mask = tf.tile(line_mask, [1, 1, 3])
        line = line * line_mask

        width_to_pad = image_width - line_width
        pad_left = width_to_pad // 2
        pad_right = width_to_pad - pad_left
        line = tf.pad(line, [[0, 0], [pad_left, pad_right], [0, 0]])
        return image + tf.cast(line, tf.uint8)

    def _preprocess_image(self, image_path):
        image = self._parse_image(image_path)
        image = self._pad_image(image)
        image = self._rotate_image(image)
        image = self._crop_vertical_stripe(image)
        has_line = tf.random.uniform(shape=[], minval=0, maxval=1) > 0.5
        if has_line:
            image = self._generate_line(image)
            label = 1  # Line present
        else:
            label = 0  # Line absent
        return image, label

    def build_pipeline(self, batch_size):
        image_files = [os.path.join(self.images_path, filename) for filename in os.listdir(self.images_path)]
        num_samples = len(image_files)

        dataset = tf.data.Dataset.from_tensor_slices(image_files)
        dataset = dataset.shuffle(num_samples)
        dataset = dataset.map(self._preprocess_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

        return dataset


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    batch_size = 1
    data_loader = DataLoader('data/without_laser/train',
                             stripe_width=150,
                             line_width_range=(4, 16),
                             length_range=(0.33, 1),
                             intensity_range=(16, 128))

    # Build the dataset pipeline
    dataset = data_loader.build_pipeline(batch_size)

    # Iterate over the dataset and plot some samples
    for images, labels in dataset:
        for i in range(batch_size):
            image = images[i].numpy().astype(int)
            label = labels[i].numpy()

            plt.imshow(image)
            plt.title("Label: {}".format(label))
            plt.axis("off")
            plt.tight_layout()
            plt.show()

        break  # Stop after the first batch
