import numpy as np
from scipy import ndimage
from scipy import stats
import matplotlib.pyplot as plt
import cv2
import icecream as ic

k = np.sqrt(2 * np.log(np.e))


def _moving_average(x, w):
    return np.convolve(x, np.ones(w), 'same') / w


def _hist_fit(element_lengths, edges):
    fit_func = stats.invgauss
    params = fit_func.fit(element_lengths)
    fit_values = fit_func.pdf(edges, *params)
    max_fit_value_index = np.where(fit_values == np.max(fit_values))[0]
    return max_fit_value_index


def _calc_sigma_from_length_distribution(element_lengths):
    if len(element_lengths) == 0:
        return 0, 0, 0
    max_value = np.max(element_lengths)
    hist, edges = np.histogram(element_lengths, bins=max_value)
    ma_size = (np.max(element_lengths) // 100) or 1
    ma = _moving_average(hist, ma_size)
    max_indicies_hist = np.where(hist == np.max(hist))
    max_indicies_ma = np.where(ma == np.max(ma))
    max_x_hist = np.round(edges[max_indicies_hist[0]])
    max_x_ma = np.round(edges[max_indicies_ma[0]])
    
    max_x_fit = _hist_fit(element_lengths, edges)

    period_hist = np.mean(max_x_hist)
    calc_sigma_hist = period_hist // (2 * np.pi)
    period_ma = np.mean(max_x_ma)
    calc_sigma_ma = period_ma // (2 * np.pi)
    period_fit = np.mean(max_x_fit)
    calc_sigma_fit = period_fit // (2 * np.pi)
    
    return calc_sigma_hist, calc_sigma_ma, calc_sigma_fit


# Макс, если будешь менять этот скрипт, то именно эта функция главное звено,
# которое используется в main и других скриптах. Связывайте, пожалуйста, через нее
def get_sigma(bin_image, mode):
    """
    Функция используется для оценки сигмы
    mode = "default", "smoothed", "gaus"
    """

    bin_image = bin_image.flatten()

    borders = bin_image[1:] != bin_image[:-1]
    borders = np.append(borders, True)
    indexes = np.where(borders)[0] + 1
    line_elements = np.split(bin_image, indexes)
    line_elements = np.array([elem.flatten() for elem in line_elements], dtype=object)[:-1]

    true_elements = filter(lambda x: x[0] == True, line_elements)
    true_element_lengths = np.array([len(elem) for elem in true_elements])

    false_elements = filter(lambda x: x[0] == False, line_elements)
    false_element_lengths = np.array([len(elem) for elem in false_elements])

    true_sigma_h, true_sigma_ma, true_sigma_fit = \
        _calc_sigma_from_length_distribution(true_element_lengths)
    false_sigma_h, false_sigma_ma, false_sigma_fit = \
        _calc_sigma_from_length_distribution(false_element_lengths)

    if mode == "default":
        return (true_sigma_h + false_sigma_h) * k
    elif mode == "smoothed":
        return (true_sigma_ma + false_sigma_ma) * k
    elif mode == "gaus":
        return (true_sigma_fit + false_sigma_fit) * k
    else:
        raise ValueError("mode most be \"default\", \"smoothed\" or \"gaus\"")
