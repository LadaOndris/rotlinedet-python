import math

import cv2
import numpy as np
from matplotlib import pyplot as plt


def pad_image(image: np.ndarray) -> np.ndarray:
    img_width = image.shape[1]
    img_height = image.shape[0]

    # Pad the image before rotation, so that the whole image is visible
    diagonal = math.ceil(math.sqrt(img_height ** 2 + img_width ** 2))
    vertical_pad = diagonal - img_height
    vertical_pad_half = vertical_pad // 2
    horizontal_pad = diagonal - img_width
    horizontal_pad_half = horizontal_pad // 2

    pad_width = [[vertical_pad_half, vertical_pad_half],
                 [horizontal_pad_half, horizontal_pad_half]]
    padded = np.pad(image, pad_width, constant_values=0)
    return padded


def convert_bgr_to_gray(image: np.ndarray) -> np.ndarray:
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return gray


def remove_extreme_intensities(image: np.ndarray, interval: int = 50) -> np.ndarray:
    median = np.median(image)
    image[image > median + interval] = median
    image[image < median - interval] = median

    hp = np.percentile(image, 95)
    ret, thres = cv2.threshold(image, thresh=hp, maxval=255, type=cv2.THRESH_BINARY)
    binary = cv2.morphologyEx(thres, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))

    replacement_value = np.percentile(image, 40)
    image[binary == 255] = replacement_value

    return image


def rotate_image(image: np.ndarray, angle_degs: float) -> np.ndarray:
    # Rotate around the centre of the image
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)

    M = cv2.getRotationMatrix2D((cX, cY), angle_degs, 1.0)
    rotated_image = cv2.warpAffine(image, M, (w, h))
    return rotated_image


def sum_columns(image: np.ndarray, min_nonzero_counts: int) -> np.ndarray:
    columns_sum = np.sum(image, axis=0).astype(float)

    # Normalize column sum by the number of nonzero intensities in that column
    nonzero_counts = np.count_nonzero(image, axis=0)
    with np.errstate(divide='ignore', invalid='ignore'):
        columns_sum /= nonzero_counts
    columns_sum[np.isnan(columns_sum)] = 0

    # Do not consider statistically insignificant cols (#pixels less than min_nonzero_counts)
    columns_sum[nonzero_counts < min_nonzero_counts] = 0

    return columns_sum


def convolve_average(array: np.ndarray, filter_size: int) -> np.ndarray:
    filter = np.ones([filter_size], dtype=float)
    filter /= np.sum(np.abs(filter))

    convolved_average = np.convolve(array, filter, mode='same')
    return convolved_average


def select_peak(array: np.ndarray):
    # Do not consider zeros in the computation of the mean
    nozeros_array = array[array > 0]

    # Select the peak that is higher than all others.
    peak_pos = np.argmax(array)

    # Metric signifies the distance from the average peak value.
    metric = array[peak_pos] / np.mean(nozeros_array)

    return peak_pos, metric


def print_line_params(peak_col, peak_value, rotation_angle, metric):
    print(
        F"Peak col: {peak_col}, rotation: {rotation_angle} [degs], peak value: {peak_value:.3f}, metric: {metric:.1f}.")


def get_peaks_at_local_extremes(orig_column_sum, averaged_column_sum, thres: float, filter_size: int) -> np.ndarray:
    # Column diff
    peaks = orig_column_sum - averaged_column_sum
    peaks[peaks < 0] = 0

    # Slopes
    slope_pixels = filter_size // 2
    dY = np.roll(averaged_column_sum, -slope_pixels, axis=0) - np.roll(averaged_column_sum, slope_pixels, axis=0)
    dY[:slope_pixels] = np.inf
    dY[-slope_pixels:] = np.inf
    slopes = np.abs(dY)

    # Filter using slopes
    slopes_mask = slopes < thres
    masked_peaks = peaks * slopes_mask
    return masked_peaks


def plot_column_sum(column_sum, averaged_column_sum) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(column_sum)
    ax.plot(averaged_column_sum)
    fig.tight_layout()
    fig.show()


def plot_peaks(peaks) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(peaks)
    fig.tight_layout()
    fig.show()


def plot_rotated_image(rotated) -> None:
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(rotated, cmap='gray')
    fig.tight_layout()
    fig.show()


def process_frame(frame: np.ndarray, rotation_step: int,
                  filter_size: int = 30, peak_slope_thres: float = 0.2,
                  min_nonzero_counts: int = 200, plot: bool = False) -> None:
    print(f"Processing frame with shape {frame.shape}.")
    gray = convert_bgr_to_gray(frame)
    # removed_outliers = remove_extreme_intensities(gray)
    padded = pad_image(gray)

    max_metric_rot = None
    max_metric = 0
    rotation_angles = np.arange(-70, 70, rotation_step)

    for angle in rotation_angles.flat:
        rotated = rotate_image(padded, angle)
        column_sum = sum_columns(rotated, min_nonzero_counts)
        averaged_column_sum = convolve_average(column_sum, filter_size)
        peaks = get_peaks_at_local_extremes(column_sum, averaged_column_sum, peak_slope_thres, filter_size)
        peak_col, metric = select_peak(peaks)

        # Note: metric is no longer used as the selection criterie
        # Instead, the raw peak value is used.
        print_line_params(peak_col, peaks[peak_col], angle, metric)
        if peaks[peak_col] > max_metric:
            max_metric = peaks[peak_col]
            max_metric_rot = angle

        if plot:
            plot_column_sum(column_sum, averaged_column_sum)
            plot_peaks(peaks)
            plot_rotated_image(rotated)

    print(f'Max metric: {max_metric} at {max_metric_rot}°')
