import pandas as pd
import os
import h5py


SCRIPT_PATH = os.path.dirname(os.path.abspath(__file__))


def save(orig_phantom, processed_phantom, porosity, blobns, noise_info, num_of_angles, tag='train'):
    '''
    Saves phantom to the database.
        
    Parameters:
    -----------
    orig_phantom: ndarray.
        Original phantom

    processed_phantom: ndarray.
        Phantom object which represents result of the experiment

    porosity: float.
        Phantom's porosiity
    
    blobns: int.
        Phantom's blobiness
    
    noise_info: float or str.
        Information about the noise (e.g. probability)
    
    num_of_angles: int.
        Number of Radon projections
    
    results:
    --------
    out: phantoms.h5 file
        See in the  script's directory in the folder 'database'
    '''

    dimension = len(orig_phantom.shape)
    save_path = os.path.join(SCRIPT_PATH, 'database')

    try:
        os.mkdir(save_path)
    except OSError:
        print("changing existed file")
    save_path = os.path.join(save_path, 'phantoms.h5')
    
    with h5py.File(save_path, 'a') as hdf:
        ps = porosity, blobns, noise_info, num_of_angles

        try:
            current_id_values = list(hdf.get(f'{dimension}_dimensional'))
            for id_value in current_id_values:
                if tuple(hdf.get(f'{dimension}_dimensional').get(id_value).attrs.values()) == ps:
                    id_indx = id_value
                    print("editing file already exists")
                    break
                else:
                    id_indx = str(int(id_value) + 1)
        except BaseException:
            id_indx = '1'
        
        phantom_root_path = f'{dimension}_dimensional/{id_indx}'
        phantom_root_tag_path = f'{phantom_root_path}/{tag}'
        try:
            phantom_file_group = hdf.create_group(phantom_root_tag_path)
        except BaseException:
            phantom_file_group = hdf.get(phantom_root_tag_path)
        
        phantom_root_group = hdf.get(phantom_root_path)
        phantom_root_group.attrs["porosity"] = porosity
        phantom_root_group.attrs["blobiness"] = blobns
        phantom_root_group.attrs["noise"] = noise_info
        phantom_root_group.attrs["num_of_angles"] = num_of_angles
        
        try:
            phantom_file_group.create_dataset("orig_phantom", data = orig_phantom)
            phantom_file_group.create_dataset("processed_phantom", data = processed_phantom)
        except BaseException:
            del phantom_file_group["orig_phantom"]
            del phantom_file_group["processed_phantom"]
            phantom_file_group.create_dataset("orig_phantom", data = orig_phantom)
            phantom_file_group.create_dataset("processed_phantom", data = processed_phantom)


def show_data_info():
    '''
    shows table with existed phantom content: dimension, id_inx, tags,
    and phanrom attributes: porosity, number of angles, blobiness, noise info

    Returns:
    --------
    out: pandas.core.frame.DataFrame
        Table format. Use .head() to see first 5 rows.
    ''' 
    open_path = os.path.join(SCRIPT_PATH, 'database', 'phantoms.h5')

    df = pd.DataFrame()
    for dim in [2, 3]:
        try:
            with h5py.File(open_path, 'r') as hdf:
                id_indxes = list(hdf.get(f'{dim}_dimensional'))

                porosities, blobnses, num_of_angles, noises, tags = [], [], [], [], []

                for id_indx in id_indxes:
                    tags.append(list(hdf.get(f'{dim}_dimensional').get(str(id_indx))))
                    p, b, n, a = list(hdf.get(f'{dim}_dimensional').get(id_indx).attrs.values())
                    porosities.append(p)
                    blobnses.append(b)
                    num_of_angles.append(a)
                    noises.append(n)
                dim_data = {
                    'dimension': dim,
                    'id_indx': id_indxes,
                    'porositiy': porosities,
                    'blobiness': blobnses,
                    'num_of_angles': num_of_angles,
                    'noise': noises,
                    'tags (tring array)': tags}
                dim_df = pd.DataFrame(dim_data)
                df = df.append(dim_df, ignore_index = True)
        except BaseException:
            print(f'files with dimension {dim} does not exist')

    return df


def add_csv(dimension:int,
            id_indx: int, 
            tag:str, 
            csv_file: pd.core.frame.DataFrame):
    '''
    Use this function to save the csv file for ML process.
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
    
    csv_file: pandas.core.frame.DataFrame.
        pandas dataframe that you want to save as csv
    
    results:
    --------
    out: {dimension}_{id_indx}_{tag}.csv.csv file
        See in the  script's directory in the folder 'database'
    '''

    csv_name = f'{dimension}_{id_indx}_{tag}.csv'
    save_path = os.path.join(SCRIPT_PATH, 'database', csv_name)
    csv_file.to_csv(save_path, index = False)


def get_data(dimension: int, 
            id_indx: int, 
            tag: str, 
            what_to_return: str):
    '''
    Use this function to get needed data for ML process.
    Use show_data_info function to find out dimension, id_inx and tags, which exist.

    Parameters:
    -----------
    dimension: 2 or 3. 
        Dimension of phantom Euclidian space

    id_indx: 1,2,3,etc.
        id for phantom with certain porosity, blobiness and experiment parameters

    tag: 'test', 'train' or another
        This parameter controls conflicts if several csv files are generated for 1 phanom.

    what_to_return: 'csv', 'orig_phantom', 'processed_phantom'.
        'csv' - returns csv file with anformation about pixel values, their neigbours, etc
        'orig_phantom' - returns of original phantom
        'processed_phantom' - returns ndarray of processed phantom (experiment conducted)

    Returns:
    ----------
    out: pandas.core.frame.DataFrame or ndarray.
        Depends on what_to_return parameter
    '''

    open_path = os.path.join(SCRIPT_PATH, 'database')
    dataset = None
    if not what_to_return == "csv":
        open_path = os.path.join(open_path, 'phantoms.h5')
        with h5py.File(open_path, 'r') as hdf:
            dataset = hdf.get(f'{dimension}_dimensional').get(str(id_indx)).get(tag).get(what_to_return)
            dataset = dataset.value
    else:
        csv_name = f'{dimension}_{id_indx}_{tag}.csv'
        open_path = os.path.join(open_path, csv_name)
        dataset = pd.read_csv(open_path)

    return dataset


