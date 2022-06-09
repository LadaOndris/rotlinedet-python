from unittest import TestCase

import numpy as np

from rotation import get_shear_shifts


class Test(TestCase):

    def degs2rads(self, angle_degs: int) -> float:
        return angle_degs / 180 * np.pi

    def test_get_shear_shifts_45_degrees(self):
        img_height = 100
        expected_shifts = np.arange(img_height)

        shifts = get_shear_shifts(self.degs2rads(45), img_height)

        np.testing.assert_almost_equal(shifts, expected_shifts)

    def test_get_shear_shifts_90_degrees(self):
        img_height = 100
        expected_shifts = np.zeros([img_height])

        shifts = get_shear_shifts(self.degs2rads(90), img_height)

        np.testing.assert_almost_equal(shifts, expected_shifts)
