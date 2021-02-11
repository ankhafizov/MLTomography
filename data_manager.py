import os
import pandas as pd

SCRIPT_PATH = os.path.dirname(os.path.abspath(__file__))
DATA_FOLDER_NAME = 'data_frames'
EVRAZ_DATA = "evraz_data.csv"
DEFAULT_SEP = ";"
DEFAULT_DECIMAL = ","


def load_data(csv_file_name=EVRAZ_DATA):
    """ загружает csv файл
    csv_file_name : str defaut "evraz_data.csv"
        Название csv файла
    
    Returns
    -------
    out : pandas.Dataframe
        таблица данных
    """
    file_path = os.path.join(SCRIPT_PATH, DATA_FOLDER_NAME, csv_file_name)

    return pd.read_csv(file_path, sep=DEFAULT_SEP, decimal=DEFAULT_DECIMAL)


def save_data(df, file_name):
    """ сохраняет csv файл
    df : pandas.Dataframe
        таблица данных
    file_name : str
        Название csv или xlsx файла в который нужно сохранить df
    
    Returns
    -------
    """

    file_path = os.path.join(SCRIPT_PATH, DATA_FOLDER_NAME, file_name)

    if file_name == EVRAZ_DATA:
        raise ValueError(f"Name file_name={EVRAZ_DATA} constricted")
    if os.path.exists(file_path):
        os.remove(file_path)

    if file_name[-3:]=="csv":
        df.to_csv(file_path, sep=DEFAULT_SEP, decimal=DEFAULT_DECIMAL)
    elif file_name[-3:]=="lsx":
        df.to_excel(file_path)
    else:
        raise ValueError(f"file_name must consist .csv or .xlsx")