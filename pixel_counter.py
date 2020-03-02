import numpy as np
import opener
import pandas as pd
import os


def count_neighbor_average(arr, centre):
    dim = len(arr.shape)
    
    sum_neighbour = 0
    count = 0
    
    if dim == 2:
        for i in [-1, 0, 1]:
            for j in [-1, 0, 1]:
                if (i,j)!=(0,0):
                    ind = np.asarray(centre) + np.asarray((i,j))
                    if np.all(ind>=0):
                        try:
                            point = arr[ind[0]][ind[1]]
                            sum_neighbour += point
                            count += 1
                        except IndexError:
                            point = 0
        average = sum_neighbour / count
    else:
        for i in [-1, 0, 1]:
            for j in [-1, 0, 1]:
                for k in [-1, 0, 1]:
                    if (i,j,k)!=(0,0,0):
                        ind = np.asarray(centre) + np.asarray((i,j,k))
                        if np.all(ind>=0):
                            try:
                                point = arr[ind[0]][ind[1]][ind[2]]
                                sum_neighbour += point
                                count += 1
                            except IndexError:
                                point = 0
        average = sum_neighbour / count
    
    return average


def count_neighbor_average_array(img):
    dim = len(img.shape)
    average_array = np.zeros(img.shape)
    
    if dim == 2:
        for i, line in enumerate(img):
            for j in range(len(line)):
                average_array[i][j] = count_neighbor_average(img, (i,j))
    else:
        for i, planar in enumerate(img):
            for j, line in enumerate(planar):
                for k in range(len(line)):
                    average_array[i][j][k] = count_neighbor_average(img, (i,j,k))
    return average_array


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

