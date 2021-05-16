import numpy as np

import MLTomography.wavelet_map_generator as wmp


def get_sigma(img, porosity):
    wavelet_width = wmp.get_wavelet_width_for_2d_image(img)

    k = [-3.29261187,  4.3127627,  -1.90858856,  0.9049531 ]
    porosity = np.min([porosity, 1-porosity])
    return 0.63*wavelet_width
