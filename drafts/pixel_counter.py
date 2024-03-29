import data_manager as dm
from scipy import ndimage
import numpy as np
import pandas as pd


def count_neighbor_average_array(arr: np.ndarray):
    '''
    For each single pixel counts average intensity value of it surroundings (adjacent pixels).

    Parameters:
    -----------
    arr: ndarray.
        Array with pixel intensities (e.g. PoreSpy phantom images)
    
    Results:
    -----------
    out: ndarray
        array with the average values
    '''

    dim = len(arr.shape)
    number, kern = (26, [[[0]]]) if dim == 3 else (8, [[0]])

    kern = np.pad(kern, (1,1), constant_values=(1, 1))
    av_arr = ndimage.convolve(arr, kern)/number
    
    return av_arr


def count_neighbor_average_array_and_save(dimension:int,
                                        id_indx: int, 
                                        tag:str):
    '''
    Use this function to count neighbour pixels' average intensities and save the csv 
    file for ML process.
    Use show_data_info function to find out dimension, id_inx and tags, which exist.

    Parameters:
    -----------
    dimension: 2 or 3. 
        Dimension of phantom Euclidian space

    id_indx: 1,2,3,etc.
        id for phantom with certain porosity, blobiness and experiment parameters

    tag: 'test', 'train' or another
        This parameter controls conflicts if several csv files are generated for 1 phanom.
        Keep it different for staging different images with similar parameters
    
    results:
    --------
    out: csv file
        See in the  script's directory in the folder 'database'
        To open use data_manager.get_data(args) function
    '''

    orig_phantom = dm.get_data(dimension, id_indx, tag, 'orig_phantom')
    proc_phantom = dm.get_data(dimension, id_indx, tag, 'processed_phantom')

    neighbor_pixel_average = count_neighbor_average_array(proc_phantom)

    data = {'neighbor_average': np.concatenate(neighbor_pixel_average),
            'proc_phantom_pixel_values': np.concatenate(proc_phantom),
            'pixel_real_value': np.concatenate(orig_phantom)}
    csv_file = pd.DataFrame(data)

    dm.add_csv(dimension, id_indx, tag, csv_file)

