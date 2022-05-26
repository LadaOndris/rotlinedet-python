
import numpy as np

def convert_rgb_to_gray(image) -> np.ndarray:
    pass

def remove_extreme_intensities(image) -> np.ndarray:
    pass

def rotate_image(image, angle) -> np.ndarray:
    pass

def sum_columns(image) -> np.ndarray:
    pass

def convolve(array) -> np.ndarray:
    pass

def select_peaks(array):
    pass

def print_line_params(peaks, rotation_angle):
    pass

def process_frame(frame, rotation_step: int) -> None:
    gray = convert_rgb_to_gray(frame)
    removed_outliers = remove_extreme_intensities(gray)
    
    rotation_angles = np.arange(-90, 90, rotation_step)
    for angle in range(rotation_angles.flat):
        rotated = rotate_image(removed_outliers, angle)
        column_sum = sum_columns(rotated)
        convolved = convolve(column_sum)
        peaks = select_peaks(convolved)
        print_line_params(peaks, angle)
    