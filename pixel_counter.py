import numpy as np
import opener
import pandas as pd
import os
from scipy import ndimage


def count_neighbor_average_array(arr):
    dim = len(arr.shape)
    number, kern = (26, [[[0]]]) if dim == 3 else (8, [[0]])
    
    kern = np.pad(kern, (1,1), constant_values=(1, 1))
    filt_arr = ndimage.convolve(arr, kern)/number
    
    return filt_arr


def save_data(phantom_file_key, tag):
    phantoms = opener.open(phantom_file_key)
    orig_phantom = phantoms['original']
    proc_phantom = phantoms['processed']

    neighbor_pixel_average = count_neighbor_average_array(proc_phantom)
    data = {'neighbor_average': np.concatenate(neighbor_pixel_average),
            'proc_phantom_pixel_values': np.concatenate(proc_phantom),
            'pixel_real_value': np.concatenate(orig_phantom)}
    df = pd.DataFrame(data)

    save_path = os.path.dirname(os.path.realpath(__file__))+f'\\__pycache__\\{tag}_pixel_dataframe.csv'
    df.to_csv(save_path)

