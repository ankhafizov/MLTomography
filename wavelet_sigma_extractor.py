import numpy as np

import MLTomography.wavelet_map_generator as wmp


def get_sigma(img, porosity, mode="simple"):
    """
    mode="simple"/"polynomial"
    """
    wavelet_width = wmp.get_wavelet_width_for_2d_image(img)

    if mode=="simple":
        empirical_constant = 0.63
        return empirical_constant*wavelet_width
    elif mode=="polynomial":
        coefs = [-3.29261187,  4.3127627,  -1.90858856,  0.9049531]
        p = np.min([porosity, 1-porosity])
        return np.array(coefs) @ np.array([p**3, p**2, p, 1])
