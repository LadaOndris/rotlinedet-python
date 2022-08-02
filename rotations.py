import math
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
from numba import njit, prange


@njit
def get_shear_shifts(angle_rads: float, img_height: int) -> np.ndarray:
    shift_coeff = 1 / math.tan(angle_rads)
    ys = np.arange(img_height)
    shifts = ys * shift_coeff
    return shifts


@njit(parallel=True)
def shift_rows(image, shifts) -> np.ndarray:
    for row in prange(shifts.shape[0]):
        image[row, :] = np.roll(image[row, :], shifts[row])
    return image


def shear_image_opencv(image, angle):
    (h, w) = image.shape[:2]
    shear = math.tan(angle)
    M = np.float64([[1, shear, 0],
                    [0, 1, 0]])
    sheared_image = cv2.warpAffine(image, M, (w, h))
    return sheared_image


def shear_image(image: np.ndarray, angle_degs: float) -> np.ndarray:
    assert -90 <= angle_degs <= 90
    shear_angle = angle_degs / 180 * np.pi

    change_threshold = np.pi / 2 - math.atan(image.shape[0] / image.shape[1])
    if abs(shear_angle) > change_threshold:
        image = np.rot90(image)
        shear_angle = np.pi / 2 - shear_angle

    # shear_angle = np.pi / 2 - shear_angle
    img_height = image.shape[0]

    shifts = get_shear_shifts(np.pi / 2 - shear_angle, img_height).astype(int)

    max_shift = shifts[-1]
    if max_shift < 0:
        pad_x = [-max_shift, 0]
    else:
        pad_x = [0, max_shift]
    image = np.pad(image, [[0, 0], pad_x])

    # image = shift_rows(image, shifts)
    image = shear_image_opencv(image, shear_angle)

    return image


if __name__ == "__main__":
    arr = np.ones([2713, 4096], dtype=np.uint8)
    arr[2713-300:2713-200, :] = 2
    # arr_padded = np.pad(arr, [[0, 0], [4800, 4800]])
    # sheared = shear_image(arr, 30)
    st = time.time()
    sheared = shear_image(arr, 56.5)
    print((time.time() - st) * 1000)

    fig, ax = plt.subplots(figsize=(14, 7))
    ax.imshow(sheared, cmap='gray')
    fig.tight_layout()
    fig.show()
