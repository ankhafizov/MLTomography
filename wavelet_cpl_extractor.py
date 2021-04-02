import data_manager as dm
import numpy as np
import matplotlib.pyplot as plt

import wavelet_map_generator as wmp
import phantom_generator as pg


def find_y_between(x_left, x_right,
                   y_left, y_right,
                   x):
    m = (y_left-y_right)/(x_left-x_right)
    b = (x_left*y_right - x_right*y_left)/(x_left-x_right)
    return m*x+b


def find_closest_values_in_df(map_df, column_name, value):
    unique_values = map_df[column_name].unique() 
    
    return unique_values[np.abs(unique_values - value).argsort()][:2]


def find_pore_length(map_df, porosity, wavelet_width):
    closest_porosities = find_closest_values_in_df(map_df, 'porosity', porosity)

    pore_lengths = []
    for p in closest_porosities:
        pore_length_w = []
        closest_widths = find_closest_values_in_df(map_df[map_df.porosity==p], 'wavelet_width', wavelet_width)
        for w in closest_widths:
            condition = np.logical_and(map_df.porosity==p, map_df.wavelet_width==w)
            pore_length_w.append(int(map_df.loc[condition, 'wavelet_width']))
        pore_lengths.append(find_y_between(*closest_widths, *pore_length_w, wavelet_width))
    
    pore_length = find_y_between(*closest_porosities, *pore_lengths, porosity)

    return pore_length


def extract_cpl(phantom):
    df = dm.load_dataframe("cpl_width.csv")
    porosity = np.sum(phantom) / phantom.size
    width = wmp.get_width_phantom(phantom)

    return find_pore_length(df, porosity, width)


if __name__ == "__main__":
    phantom_shape = [500, 500]
    porosity = 0.3
    cpl = 15

    phantom = ~pg.gen_phantom(phantom_shape, porosity, cpl)
    print(extract_cpl(phantom))