import data_manager as dm
import numpy as np
import matplotlib.pyplot as plt
import json

import wavelet_map_generator as wmp
import phantom_generator as pg


def _find_y_between(x_left, x_right,
                    y_left, y_right,
                    x):
    """
    linear regression [x1, y1] and [x2, y2] points
    """
    m = (y_left-y_right)/(x_left-x_right)
    b = (x_left*y_right - x_right*y_left)/(x_left-x_right)
    return m*x+b


def _find_closest_values_in_df_to_target_value(wavelet_map, column_name, target_value):
    unique_values = wavelet_map[column_name].unique() 

    return unique_values[np.abs(unique_values - target_value).argsort()][:2]


def find_sigma(wavelet_map, porosity, wavelet_width):
    closest_porosities = _find_closest_values_in_df_to_target_value(wavelet_map, 'porosity', porosity)

    sigmas = []
    for p in closest_porosities:
        wavelet_map_section = wavelet_map[wavelet_map.porosity==p]

        sigma_w = []
        closest_widths = _find_closest_values_in_df_to_target_value(wavelet_map_section,
                                                                    'wavelet_width',
                                                                    wavelet_width)
        for w in closest_widths:
            condition = np.logical_and(wavelet_map.porosity==p, wavelet_map.wavelet_width==w)
            length_series = wavelet_map.loc[condition, 'wavelet_width']
            sigma_w.append(int(length_series.to_numpy()[0]))
        sigmas.append(_find_y_between(*closest_widths, *sigma_w, wavelet_width))
    
    sigma = _find_y_between(*closest_porosities, *sigmas, porosity)

    return sigma


def extract_cpl(bin_img):
    map_file_name = json.load(open('constants.json'))["wavelet_map_name"] 
    df = dm.load_dataframe(map_file_name)
    
    porosity = np.sum(bin_img) / bin_img.size
    width = wmp.get_width_phantom(bin_img)

    return find_sigma(df, porosity, width)


if __name__ == "__main__":
    phantom_shape = [500, 500]
    porosity = 0.3
    cpl = 15

    phantom = ~pg.gen_phantom(phantom_shape, porosity, cpl)
    print(extract_cpl(phantom))