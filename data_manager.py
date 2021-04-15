import pandas as pd
import os
import h5py


SCRIPT_PATH = os.path.dirname(os.path.abspath(__file__))
DATA_FOLDER_NAME = 'dataframes'


def _add_folder(current_directory, folder_name):
    save_path = os.path.join(current_directory, folder_name)
    if not os.path.isdir(save_path):
        os.mkdir(save_path)
    return save_path


def save_dataframe(df, file_name):
    """ сохраняет csv файл
    df : pandas.Dataframe
        таблица данных
    file_name : str
        Название csv или xlsx файла в который нужно сохранить df
    
    Returns
    -------
    """
    folder_path = _add_folder(SCRIPT_PATH, DATA_FOLDER_NAME)
    file_path = os.path.join(folder_path, file_name)

    if os.path.exists(file_path):
        os.remove(file_path)

    if file_name[-3:]=="csv":
        df.to_csv(file_path, index=False)
    elif file_name[-3:]=="lsx":
        df.to_excel(file_path)
    else:
        raise ValueError(f"file_name must consist .csv or .xlsx")


def load_dataframe(csv_file_name):
    """ загружает csv файл
    csv_file_name : str defaut "train_dataframe.csv"
        Название csv файла
    
    Returns
    -------
    out : pandas.Dataframe
        таблица данных
    """
    file_path = os.path.join(SCRIPT_PATH, DATA_FOLDER_NAME, csv_file_name)

    return pd.read_csv(file_path)