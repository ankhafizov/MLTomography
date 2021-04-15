import unittest

from scipy import signal
import numpy as np

import sys, os
sys.path.append("..")

from MLTomography import wavelet_map_generator as wmg
from MLTomography.logger import logger
import MLTomography.phantom_generator as pg


class TestWaveletMapGenerator(unittest.TestCase):
    def test_get_wavelet_width_of_row_signal(self):
        t = np.linspace(0, 1, 500, endpoint=False)
        sig = signal.square(2 * np.pi * 5 * t)

        width = wmg.get_wavelet_width_of_row_signal(sig, max_width=200)
        logger.debug(f"calculated width: {width}| ground truth: 11")

        self.assertTrue(10<=width<=12)


    def test_get_wavelet_width_for_sample(self):

        porosity = 0.3
        sigma = 15
        phantom_shape = [100, 100]
        attempts=5

        phantom_width = []

        for _ in range(attempts):
            phantom = pg.gen_phantom(phantom_shape, porosity, sigma)
            phantom_width.append(wmg.get_wavelet_width_for_2d_image(phantom))

        print(np.mean(phantom_width))
        self.assertTrue(5<np.mean(phantom_width)<25)


if __name__ == '__main__':
    unittest.main()