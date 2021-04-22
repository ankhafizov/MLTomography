
import numpy as np
import scipy
import pandas as pd
import json
from progress.bar import Bar

import MLTomography.phantom_generator as pg
import MLTomography.data_manager as dm
from MLTomography.helper import invert_signal


WAVELET_TYPE = scipy.signal.ricker #mexican hat


def get_wavelet_width_of_row_signal(signal, max_width=200):
    # TODO: add optimizer to find pearson maximum
    get_pearson_corrcoef = lambda x1, x2: scipy.stats.pearsonr(x1, x2)[0]
    
    possible_widths = np.arange(1, max_width)
    wavelet_rows = scipy.signal.cwt(signal, WAVELET_TYPE, possible_widths)
    pearsons = [get_pearson_corrcoef(row, signal) for row in wavelet_rows]

    return np.argmax(pearsons)


def get_wavelet_width_for_2d_image(bin_img, axis=0):
    """
    axis =0, 1 or "all"
    """
    bin_img = np.asarray(bin_img)
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
        row = invert_signal(row) # не уверен, что это необходимо. Делаю, чтобы раскладывать именно поры по вейвлетам
        w = get_wavelet_width_of_row_signal(row)
        wavelet_widths.append(w)

    return np.median(wavelet_widths)


def get_wavelet_width_for_sample(porosity, sigma, shape):
    attempts=5

    phantom_width = []

    for _ in range(attempts):
        phantom = pg.generate_phantom(shape, porosity, sigma)
        phantom_width.append(get_wavelet_width_for_2d_image(phantom))

    return np.mean(phantom_width)


if __name__ == '__main__':
    porosities = [0.1, 0.2, 0.3, 0.4, 0.5]
    sigmas = [3, 5, 15, 30, 40, 50, 60, 70, 80, 90, 100]
    shape = (1, 1_000_000)

    map_file_name = json.load(open('constants.json'))["wavelet_map_name"] 
    
    df = pd.DataFrame(columns = ['porosity',
                                 'sigma',
                                 'wavelet_width'])

    # df = dm.load_dataframe(map_file_name)
    
    bar = Bar('Processing', max=len(sigmas)*len(porosities))

    for sigma in sigmas:
        for porosity in porosities:
            wavelet_width = get_wavelet_width_for_sample(porosity, 
                                                              sigma,
                                                              shape)

            df = df.append({'porosity': porosity,
                            'sigma': sigma,
                            'wavelet_width': wavelet_width},
                           ignore_index=True)

            dm.save_dataframe(df, map_file_name)

            bar.next()