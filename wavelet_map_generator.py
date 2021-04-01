import matplotlib.pyplot as plt
import numpy as np
import phantom_generator as pg
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


def get_width_phantom(phantom, plot_stat=True):
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
            phantom_widths.append(get_width_phantom(phantom, plot_stat=False))
    
        mean_phantom_width.append(np.mean(phantom_widths))
        std_phantom_width.append(np.std(phantom_widths))

    return mean_phantom_width, std_phantom_width


if __name__ == '__main__':
    porosities = [0.2, 0.3]
    characteristical_pore_lengths = [5, 10]
    
    phantom_shape = (50, 50)

    for porosity in porosities:
        mpw, spw = get_wavelet_widths_for_fixed_porosity(porosity, 
                                                         characteristical_pore_lengths,
                                                         phantom_shape)
        print(mpw, spw)
