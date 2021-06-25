
import numpy as np
import scipy
import pandas as pd
import json

import MLTomography.phantom_generator as pg
import MLTomography.data_manager as dm


WAVELET_TYPE = scipy.signal.ricker #mexican hat


def get_wavelet_width_of_row_signal(signal, max_width=200):
    # TODO: add optimizer to find pearson maximum
    get_pearson_corrcoef = lambda x1, x2: scipy.stats.pearsonr(x1, x2)[0]
    loss_func = lambda width: -get_pearson_corrcoef(signal, scipy.signal.cwt(signal, WAVELET_TYPE, [width])[0])

    width_optimal = scipy.optimize.minimize_scalar(loss_func, bounds=[1, 300], method='bounded').x

    return width_optimal


def get_wavelet_width_for_2d_image(bin_img, axis=0):
    """
    axis =0, 1 or "all"
    """
    
    bin_img = np.asarray(bin_img, dtype=float)
    wavelet_widths = []

    if axis == 0:
        pass
    elif axis == 1:
        bin_img = bin_img.T
    elif axis == "all":
        bin_img = np.vstack((bin_img, bin_img.T))
    else:
        raise ValueError(f"axis must be 0, 1 or \"all\", but {axis} was given")

    for row in bin_img:
        w = get_wavelet_width_of_row_signal(row)
        wavelet_widths.append(w)

    return np.median(wavelet_widths)
