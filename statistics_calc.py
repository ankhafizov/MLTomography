import numpy as np
import phantom_generator as pg
import pandas as pd
from sklearn import datasets, linear_model

from scipy.ndimage import label
from skimage.measure import regionprops


def _split_arr_to_zeros_and_ones_sizes(arr):
    """
    Splits arr into arrays lengths of ones and zeros.
    
    Parameters
    ----------
    arr: numpy.array
        binary 1 d array of 1 and 0
    
    Returns:
    --------
    segments_ones: numpy.array
        array of 1d arrays filled by ones.
    segments_zeros:
        array of 1d arrays filled by zeros.

    Example:
    --------
    if arr is [0, 0, 1, 1, 1, 0, 1, 0, 0] ==>
    segments_ones = [len([1, 1, 1]), len([1])] = [3, 1]
    segments_zeros =  [len([0, 0]), len([0]), len([0, 0])] = [2, 1, 2]
    """
    edge_indeces = np.argwhere(~(np.diff(arr) == 0)).flatten() +1
    segments = np.split(arr, edge_indeces)

    segments_zeros = []
    segments_ones = []

    for segment in segments:
        if np.any(segment):
            segments_ones.append(len(segment))
        else:
            segments_zeros.append(len(segment))

    return segments_zeros, segments_ones


def get_row_statistics(phantom, row_numbers, axis=0):
    '''
    Splits phantom image into lengths distribution of zero- and one- filled parts.

    Parameters
    ----------
    phantom: numpy.array
        binary 2d image of porous sample.
    row_numbers: int or 'all'
        number of rows randomly selected along the  axis.
    axis: 0, 1 or 'both'
        axis of 2d image of the phantom that.
    
    Returns:
    --------
    stat_ones: numpy.array
        array of 1d arrays filled by ones that corresponds to material parts splited by pores.
    stat_zeros:
        array of 1d arrays filled by zeros that corresponds to pore parts splited by material.
    
    Example:
    --------
    if phantom is [0, 0, 1, 1, 1, 0, 1, 0, 0] then for 0-axis and row_numbers='all' or row_numbers=1 ==>
    stat_ones = [3, 1]
    stat_zeros = [2, 1, 2]
    '''
    stat_ones, stat_zeros = [], []

    if axis == 0:
        if row_numbers == "all":
            row_numbers = range(phantom.shape[axis])
        for row_num in row_numbers:
            segments_zeros, segments_ones = _split_arr_to_zeros_and_ones_sizes(phantom[row_num])
            stat_zeros.append(segments_zeros)
            stat_ones.append(segments_ones)
    elif axis == 1:
        if row_numbers == "all":
            row_numbers = range(phantom.shape[axis])
        for row_num in row_numbers:
            segments_zeros, segments_ones = _split_arr_to_zeros_and_ones_sizes(phantom[:, row_num])
            stat_zeros.append(segments_zeros)
            stat_ones.append(segments_ones)
    elif axis == "both":
        if row_numbers == "all":
            row_numbers_0 = range(phantom.shape[0])
            row_numbers_1 = range(phantom.shape[1])
        for row_num in row_numbers_0:
            segments_zeros, segments_ones = _split_arr_to_zeros_and_ones_sizes(phantom[row_num])
            stat_zeros.append(segments_zeros)
            stat_ones.append(segments_ones)
        for row_num in row_numbers_1:
            segments_zeros, segments_ones = _split_arr_to_zeros_and_ones_sizes(phantom[:, row_num])
            stat_zeros.append(segments_zeros)
            stat_ones.append(segments_ones)

    stat_ones, stat_zeros = np.concatenate(stat_ones), np.concatenate(stat_zeros)

    return stat_zeros, stat_ones


def get_volume_stats(sample_volume):
    labels, _ = label(~sample_volume)
    regions=regionprops(labels)
    volumes = [r.area for r in regions if r.area>1]
    return volumes


def generate_train_data(stat_counting_function,
                        blobinesses=[1, 5, 10, 20, 30],
                        porosities=[0.1, 0.3, 0.5],
                        shape=[1000, 1000],
                        sample_count=10,
                        row_numbers="all",
                        volume_to_diameters=False,
                        axis="both"):
    train_dataframe = pd.DataFrame(columns = ['porosity', 'blobiness', 'hist_characteristical_number'])
    get_radii = lambda volumes: np.sqrt(np.asarray(volumes) / np.pi)

    for _ in range(sample_count):
        for porosity in porosities:
            for blobiness in blobinesses:
                phantom = pg.gen_phantom(shape, porosity=porosity, blobiness=blobiness)
                if stat_counting_function == get_row_stats:
                    hist_characteristical_number = np.mean(stat_counting_function(phantom, row_numbers, axis=axis)[0])
                elif stat_counting_function == get_volume_stats:
                    stats = stat_counting_function(phantom)
                    hist_characteristical_number = np.median(get_radii(stats)) if volume_to_diameters else np.median(stats)
                train_dataframe = train_dataframe.append({'porosity': porosity,
                                                          'blobiness': blobiness,
                                                          'hist_characteristical_number': hist_characteristical_number},
                                                           ignore_index=True)
    return train_dataframe


def extract_data_from_dataframe(df):
    porosities = np.unique(df["porosity"])
    blobinesses = np.unique(df["blobiness"]).astype(int)
    meshgrid = np.array(np.meshgrid(porosities, blobinesses)).T
    all_porosity_blobns_pairs = meshgrid.reshape(-1, meshgrid.shape[-1])

    train_data = []
    for p, b in all_porosity_blobns_pairs:
        diabetes_Y_train = df.loc[(df["porosity"]==p) & \
                                  (df["blobiness"]==b)]["hist_characteristical_number"].to_numpy()
        train_data += [[p, int(b), hist_characteristical_number] for hist_characteristical_number in diabetes_Y_train]
    train_data = np.asarray(train_data).T
    return train_data


def get_regression_coefs_by_plane(train_dataframe):
    train_data = extract_data_from_dataframe(train_dataframe)

    regr = linear_model.LinearRegression()
    regr.fit(train_data[0:-1].T, train_data[-1])

    porosity_coef, blobiness_coef = regr.coef_
    return regr.intercept_, porosity_coef, blobiness_coef
    

def find_blobiness(bin_image,
                   porosity,
                   stat_counting_function,
                   regression_coefs,
                   row_numbers="all",
                   axis="both",
                   volume_to_diameters=False,
                   stat_type_for_rows=0):
    if stat_counting_function == get_row_stats:
        hist_characteristical_number = np.mean(stat_counting_function(bin_image,
                                               row_numbers,
                                               axis=axis)[stat_type_for_rows])
    elif stat_counting_function == get_volume_stats:
        get_radii = lambda volumes: np.sqrt(np.asarray(volumes) / np.pi)
        stats = stat_counting_function(bin_image)
        hist_characteristical_number = np.median(get_radii(stats)) if volume_to_diameters else np.median(stats)

    return (hist_characteristical_number - regression_coefs[0] - porosity*regression_coefs[1]) / regression_coefs[2]
