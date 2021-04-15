
import numpy as np
import scipy
import pandas as pd
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


def get_wavelet_width_for_2d_image(bin_img_2d, axis=0):
    """
    axis =0, 1 or "all"
    """
    bin_img_2d = np.asarray(bin_img_2d)
    wavelet_widths = []

    if axis == 0:
        pass
    elif axis == 1:
        bin_img_2d = bin_img_2d.T
    elif axis == "all":
        bin_img_2d = np.vstack((bin_img_2d, bin_img_2d.T))
    else:
        raise ValueError(f"axis must be 0, 1 or \"all\", but {axis} was given")

    for row in bin_img_2d:
        row = invert_signal(row) # не уверен, что это необходимо. Делаю, чтобы раскладывать именно поры по вейвлетам
        w = get_wavelet_width_of_row_signal(row)
        wavelet_widths.append(w)

    return np.median(wavelet_widths)


if __name__ == '__main__':
    df = pd.DataFrame(columns = ['porosity',
                                 'characteristical_pore_length',
                                 'wavelet_width',
                                 'wavelet_width_std'])
    df = dm.load_dataframe("cpl_width.csv")
    porosities = [0.1, 0.2, 0.3, 0.4, 0.5]
    characteristical_pore_lengths = [3, 5, 15, 30, 40, 50, 60, 70, 80, 90, 100]
    
    phantom_shape = (500, 500)
    bar = Bar('Processing', max=len(porosities)*len(characteristical_pore_lengths))

    for porosity in porosities:
        widths, std_widths = get_wavelet_widths_for_fixed_porosity(porosity, 
                                                                   characteristical_pore_lengths,
                                                                   phantom_shape)
        for cpl, w, std_w in zip(characteristical_pore_lengths, widths, std_widths):
            df = df.append({'porosity': porosity,
                            'characteristical_pore_length': cpl,
                            'wavelet_width': w,
                            'wavelet_width_std': std_w}, ignore_index=True)

            dm.save_dataframe(df, "cpl_width.csv")

            bar.next()