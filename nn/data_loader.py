import glob
import math

import tensorflow as tf
import tensorflow_addons as tfa
from matplotlib import pyplot as plt


class DataLoader:
    """
    Loads images from a folder defined by images_path and
    creates a tf.data pipeline. All images are divided in subdirectories
    according to the intensity of the line drawn in the image.
    The folder has the format `i{intensity_value}`. Each image file name
    is of the following format: `w{width}_l{length}_x{point1_x}_y{point1_y}_x{point2_x}_y{point2_y}_{index}.jpg`.
    The two points define the start and end pixel of the line in the original image.
    """

    def __init__(self, images_path: str,
                 stripe_width: int,
                 target_image_height: int):
        """
        :param images_path: Path to a folder with images that should be used for generating the dataset.
        :param stripe_width: The width of the cropped vertical stripe from the rotated image.
        :param target_image_height: Determines the height of the image.
        """
        self.images_path = images_path
        self.stripe_width = stripe_width
        self.target_image_height = target_image_height

    def _parse_image_filename(self, filepath):
        filename = tf.strings.split(filepath, '/')[-1]
        parts = tf.strings.split(filename, '_')
        parameters = {
            'width': tf.strings.to_number(self._substring_from(parts[0], 1), out_type=tf.int32),
            'length': tf.strings.to_number(self._substring_from(parts[1], 1), out_type=tf.int32),
            'point1_x': tf.strings.to_number(self._substring_from(parts[2], 2), out_type=tf.int32),
            'point1_y': tf.strings.to_number(self._substring_from(parts[3], 2), out_type=tf.int32),
            'point2_x': tf.strings.to_number(self._substring_from(parts[4], 2), out_type=tf.int32),
            'point2_y': tf.strings.to_number(self._substring_from(parts[5], 2), out_type=tf.int32),
            'img_width': tf.strings.to_number(parts[6], out_type=tf.int32),
            'img_height': tf.strings.to_number(parts[7], out_type=tf.int32),
            'filepath': filepath
        }
        # Angle to rotate it into horizontal position
        rotation_angle = tf.atan2(tf.cast(parameters['point2_y'] - parameters['point1_y'], tf.float32),
                                  tf.cast(parameters['point2_x'] - parameters['point1_x'], tf.float32))
        # Move to vertical from horizontal
        parameters['angle'] = rotation_angle + math.pi / 2
        return parameters

    def _substring_from(self, string, pos):
        return tf.strings.substr(string, pos, tf.strings.length(string))

    def _load_image(self, parsed):
        image = tf.io.read_file(parsed['filepath'])
        image = tf.image.decode_jpeg(image, channels=3)
        return image

    def _pad_image(self, image, parsed):
        img_height = tf.shape(image)[0]
        img_width = tf.shape(image)[1]

        # Pad the image before rotation, so that the whole image is visible
        diagonal = tf.math.sqrt(tf.math.square(tf.cast(img_height, tf.float32)) +
                                tf.math.square(tf.cast(img_width, tf.float32)))
        diagonal = tf.cast(tf.math.ceil(diagonal), tf.int32)
        vertical_pad = diagonal - img_height + 110
        vertical_pad_half = vertical_pad // 2
        horizontal_pad = diagonal - img_width + 110
        horizontal_pad_half = horizontal_pad // 2

        pad_values = [[vertical_pad_half, vertical_pad - vertical_pad_half],
                      [horizontal_pad_half, horizontal_pad - horizontal_pad_half],
                      [0, 0]]
        padded = tf.pad(image, pad_values, constant_values=0)

        # Translate the points
        parsed['point1_x'] += horizontal_pad_half
        parsed['point2_x'] += horizontal_pad_half
        parsed['point1_y'] += vertical_pad_half
        parsed['point2_y'] += vertical_pad_half
        return padded, parsed

    def _rotate_image(self, image, parsed):
        rotation_angle = parsed['angle']
        generate_no_line_sample = tf.random.uniform(shape=[], minval=0, maxval=1) > 1.5
        if generate_no_line_sample:
            rotation_angle = tf.random.uniform(shape=[], minval=0, maxval=2 * 3.14159)
            label = 0  # Line absent
        else:
            label = 1  # Line present
        rotated_image = tfa.image.rotate(image, rotation_angle)

        image_center = tf.cast(tf.shape(image)[:2] / 2, tf.int32)
        point = tf.stack([parsed['point2_x'], parsed['point2_y']])
        rotated_point = self.rotate_point_around_center(point, image_center, parsed['angle'])
        parsed['point2_x'] = rotated_point[0]
        parsed['point2_y'] = rotated_point[1]

        point = tf.stack([parsed['point1_x'], parsed['point1_y']])
        rotated_point = self.rotate_point_around_center(point, image_center, parsed['angle'])
        parsed['point1_x'] = rotated_point[0]
        parsed['point1_y'] = rotated_point[1]

        return rotated_image, label, parsed

    def _crop_stripe(self, image, parsed):
        """
        Crops a stripe around the vertical line.
        Determines the target stripe width and the necessary size to be cropped
        on the left and right size of the vertical line.
        It should be an appropriate length such that after it the stripe
        height is resized to `target_image_height`, the stripe width
        will be `stripe_width` (ratio needs to be maintained).
        """
        line_x = tf.cast(parsed['point2_x'], tf.int32)
        img_height = tf.shape(image)[0]

        # Determine the stripe width (before image resizing)
        factor = tf.cast(img_height, tf.float32) / self.target_image_height
        target_stripe_width = tf.cast(self.stripe_width * factor, tf.int32)
        half_stripe_width = tf.cast(tf.round(target_stripe_width // 2), tf.int32)

        # Crop the vertical stripe
        offset_width = line_x - half_stripe_width
        offset_height = 0  # Crop the whole vertical portion
        target_height = img_height
        target_width = target_stripe_width
        cropped_image = tf.image.crop_to_bounding_box(image, offset_height, offset_width,
                                                      target_height, target_width)
        return cropped_image

    def rotate_point_around_center(self, point, center, angle):
        centered_point = point - center
        rotated_centered_point = self._rotate_point(centered_point, angle)
        rotated_point = tf.cast(rotated_centered_point, tf.int32) + center
        return rotated_point

    def _rotate_point(self, point, angle):
        """
        Rotate a 2D point by a specified angle clockwise.

        :param point: A tensor representing the 2D point with shape (2,).
        :param angle: The angle of rotation in radians.
        :return: The rotated point.
        """
        x = tf.cast(point[0], tf.float32)
        y = tf.cast(point[1], tf.float32)
        cos_angle = tf.cos(angle)
        sin_angle = tf.sin(angle)
        rotated_x = x * cos_angle + y * sin_angle
        rotated_y = - x * sin_angle + y * cos_angle
        rotated_point = tf.stack([rotated_x, rotated_y])
        return rotated_point

    def _resize_stripe(self, image):
        resized_image = tf.image.resize(image, [self.target_image_height, self.stripe_width])
        return resized_image

    def _preprocess_image(self, filepath):
        parsed = self._parse_image_filename(filepath)
        image = self._load_image(parsed)
        image, parsed = self._pad_image(image, parsed)
        rotated_image, label, parsed = self._rotate_image(image, parsed)
        cropped_image = self._crop_stripe(rotated_image, parsed)
        resized_image = self._resize_stripe(cropped_image)
        return (rotated_image, resized_image), label

    def build_pipeline(self, batch_size: int):
        image_files = glob.glob(self.images_path)
        dataset = tf.data.Dataset.from_tensor_slices(image_files)
        dataset = dataset.shuffle(len(image_files))
        dataset = dataset.map(self._preprocess_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
        return dataset


def plot_image(img, label):
    image = img.numpy().astype(int)
    label = label.numpy()

    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    plt.title("Label: {}".format(label))
    plt.axis("off")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    images_path = 'data/testdataset/i128/*'
    stripe_width = 32
    target_image_height = 640
    batch_size = 4

    loader = DataLoader(images_path, stripe_width, target_image_height)
    dataset = loader.build_pipeline(batch_size)

    for images, labels in dataset:
        for i in range(batch_size):
            plot_image(images[0][i], labels[i])
            plot_image(images[1][i], labels[i])
        break  # Stop after the first batch
