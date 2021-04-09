
import phantom_generator as pg
import data_manager as dm
import pandas as pd

import matplotlib.pyplot as plt
import numpy as np
from scipy import signal, stats

WAVELET = signal.ricker


def invert_signal(sig):
    sig = sig.astype(int) if sig.dtype==bool else sig
    return sig.max() - sig


def get_width_of_a_row(sig):
    widths = np.arange(1, 50)
    cwtmatr = signal.cwt(sig, WAVELET, widths)
    pearsons = [stats.pearsonr(row, sig)[0] for row in cwtmatr]
    plt.plot(pearsons)

    return np.argmax(pearsons)


def get_width_phantom(phantom):
    """
    прохожу по всем рядам и возвращаю медиану
    """

    widths = []

    # все ряды по горизонтали
    for row in phantom:
        w = get_width_of_a_row(invert_signal(row))
        widths.append(w)

    # все ряды по вертикали если есть анизатропия (закоментил для ускорения рассчетов)
    # for row in phantom.T:
    #     w = get_width_of_a_row(invert_signal(row))
    #     widths.append(w)

    return np.median(widths)


def get_wavelet_widths_for_fixed_porosity(porosity,
                                          characteristical_pore_lengths,
                                          phantom_shape,
                                          attempts=5):

    mean_phantom_width, std_phantom_width = [], []

    for cpl in characteristical_pore_lengths:
        phantom_widths = [] 
        for _ in range(attempts):
            phantom = pg.gen_phantom(phantom_shape, porosity, cpl)
            phantom_widths.append(get_width_phantom(phantom))
    
        mean_phantom_width.append(np.mean(phantom_widths))
        std_phantom_width.append(np.std(phantom_widths))

    return mean_phantom_width, std_phantom_width


if __name__ == '__main__':
    df = pd.DataFrame(columns = ['porosity',
                                 'characteristical_pore_length',
                                 'wavelet_width',
                                 'wavelet_width_std'])
    porosities = [0.1, 0.2, 0.3, 0.4, 0.5]
    characteristical_pore_lengths = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    
    phantom_shape = (500, 500)

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