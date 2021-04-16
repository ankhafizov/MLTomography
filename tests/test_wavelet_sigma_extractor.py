import unittest

from scipy import signal
import numpy as np

import sys, os
sys.path.append("..")

from MLTomography import wavelet_sigma_extractor as wse
from MLTomography.logger import logger
import MLTomography.phantom_generator as pg

import icecream as ic

class TestWaveletSigmaExtractor(unittest.TestCase):
    def test_get_sigma(self):
        shape=[300, 300]
        porosity=0.5
        sigma=15

        phantom = pg.generate_phantom(shape, porosity, sigma)
        sigma_extracted = wse.get_sigma(phantom, porosity)
        ic.ic(sigma_extracted)

        self.assertTrue(10<=sigma_extracted<=20)


if __name__ == '__main__':
    unittest.main()
