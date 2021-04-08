import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt
import cv2


# %%
def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'same') / w


def sigma_estimate(size=10_000_000, sigma=1):

    range = np.arange(size)
    noise_image = np.random.random(size)
    image = ndimage.gaussian_filter(noise_image, sigma=sigma, truncate=4)
    bin_image = image >= 0.5

    borders = bin_image[1:] != bin_image[:-1]
    borders = np.append(borders, True)
    indexes = np.where(borders)[0] + 1
    line_elements = np.split(borders, indexes)
    element_lengths = np.array([len(elem) for elem in line_elements])[:-1]

    hist, edges = np.histogram(element_lengths, bins=np.max(element_lengths))
    ma_size = (np.max(element_lengths) // 100) or 1
    ma = moving_average(hist, ma_size)
    max_indicies_hist = np.where(hist == np.max(hist))
    max_indicies_ma = np.where(ma == np.max(ma))
    max_x_hist = np.round(edges[max_indicies_hist[0]])
    max_x_ma = np.round(edges[max_indicies_ma[0]])

    period_hist = np.mean(max_x_hist) * 2
    calc_sigma_hist = period_hist // (2 * np.pi)
    period_ma = np.mean(max_x_ma) * 2
    calc_sigma_ma = period_ma // (2 * np.pi)
    
#     csh = np.round(calc_sigma_hist * np.sqrt(2))
#     csm = np.round(calc_sigma_ma * np.sqrt(2)) # why np.sqrt(2)?
    
#     csh = np.round(calc_sigma_hist * np.sqrt(2 * np.log(2)))
#     csm = np.round(calc_sigma_ma * np.sqrt(2 * np.log(2))) # why np.sqrt(2 * np.log(2))?

    return calc_sigma_hist, calc_sigma_ma#, csh, csm


x = []
y_h = []
y_m = []
y_h_1 = []
y_m_1 = []
y_h_2 = []
y_m_2 = []
k1 = np.sqrt(2 * np.log(2))
k2 = np.sqrt(2)


def processing_sigma(sigma):
    sigma_e = sigma_estimate(sigma=sigma)
    x.append(sigma)
    y_h.append(sigma_e[0])
    y_m.append(sigma_e[1])
    y_h_1.append(sigma_e[0] * k1)
    y_m_1.append(sigma_e[1] * k1)
    y_h_2.append(sigma_e[0] * k2)
    y_m_2.append(sigma_e[1] * k2)
    print(f'sigma: {sigma}, calc: {sigma_e}')


for sigma in np.arange(1, 10, 1, dtype=np.int):
    processing_sigma(sigma)
    
for sigma in np.arange(10, 101, 10, dtype=np.int):
    processing_sigma(sigma)


plt.figure(figsize=(10, 10))
plt.plot(x, x, color='gray')
plt.scatter(x, y_h, color='blue')
plt.scatter(x, y_m, color='red')
plt.scatter(x, y_h_1, color='blue', marker='+')
plt.scatter(x, y_m_1, color='red', marker='+')
plt.scatter(x, y_h_2, color='blue', marker='x')
plt.scatter(x, y_m_2, color='red', marker='x')


def calc_sigma(element_lengths):
    if len(element_lengths) == 0:
        print(f'empty element_lengths')
        return 0, 0
    max_value = np.max(element_lengths)
    hist, edges = np.histogram(element_lengths, bins=max_value)
    ma_size = (np.max(element_lengths) // 100) or 1
    ma = moving_average(hist, ma_size)
    max_indicies_hist = np.where(hist == np.max(hist))
    max_indicies_ma = np.where(ma == np.max(ma))
    max_x_hist = np.round(edges[max_indicies_hist[0]])
    max_x_ma = np.round(edges[max_indicies_ma[0]])

    period_hist = np.mean(max_x_hist)
    calc_sigma_hist = period_hist // (2 * np.pi)
    period_ma = np.mean(max_x_ma)
    calc_sigma_ma = period_ma // (2 * np.pi)
    
    return calc_sigma_hist, calc_sigma_ma


k = np.sqrt(2 * np.log(np.e))

def sigma_estimate_2(bin_image):

    borders = bin_image[1:] != bin_image[:-1]
    borders = np.append(borders, True)
    indexes = np.where(borders)[0] + 1
    line_elements = np.split(bin_image, indexes)
    line_elements = np.array([elem.flatten() for elem in line_elements], dtype=object)[:-1]

    true_elements = filter(lambda x: x[0] == True, line_elements)
    true_element_lengths = np.array([len(elem) for elem in true_elements])

    false_elements = filter(lambda x: x[0] == False, line_elements)
    false_element_lengths = np.array([len(elem) for elem in false_elements])
    
    if len(true_element_lengths) == 0 or len(false_element_lengths) == 0:
        print(f'line_elements: {line_elements}')
        print(f'true_elements: {true_elements}')
        print(f'false_elements: {false_elements}')

    true_sigma_h, true_sigma_ma = calc_sigma(true_element_lengths)
    false_sigma_h, false_sigma_ma = calc_sigma(false_element_lengths)
    
    # return true_sigma_h + false_sigma_h, true_sigma_ma + false_sigma_ma
    return (true_sigma_ma + false_sigma_ma) * k