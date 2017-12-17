import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

train_path = "train.json/data/processed/train.json"
img_size = 75




def load_data(path = train_path):
    """
    Loads the data from .json format into a pd.DataFrame

    Args:
    path -- File path to data.

    Returns:
    data -- A `pd.DataFrame` with columns:
            band_1  |  band_2  |  id  | inc_angle  |  is_iceberg
    """
    data = pd.read_json(path, orient="records")
    return data


def get_row(data, index = None, img_id = None):
    """
    Returns formatted row from data base either on
    row index or row id

    Args:
    data -- A `pd.DataFrame`.
    index -- A `int`. Row index of image in data.
    img_id -- A `string`. Element of `id` column of `data`.

    Returns:
    row -- A `pd.Series` with keys:
           ['band_1', 'band_2', 'id', 'inc_angle', 'is_iceberg']
    """
    if index == None:
        if img_id not in data.values:
            raise IndexError("Provided img_id is not in data.")
        index = data.id[data.id == img_id].keys()[0]

    row = data.iloc[index, :]
    return row


def plot_img(data, band = 1, index = None, img_id = None):
    """
    Plots image of ship/iceberg

    Args:
    data -- A `pd.DataFrame`, the result of load_data()
    band -- A `int`. Which band to plot:
            1 -- band_1
            2 -- band_2
            0 -- both
    index -- A `int`. Row index of image in data.
    img_id -- A `string`. Element of `id` column of `data`.

    Returns:

    """
    row = get_row(data, index, img_id)
    band1 = np.array(row["band_1"]).reshape(img_size, img_size)
    band2 = np.array(row["band_2"]).reshape(img_size, img_size)

    if band == 1:
        plt.figure(figsize = (3,3))
        plt.imshow(band1)
    elif band == 2:
        plt.figure(figsize = (3,3))
        plt.imshow(band2)
    elif band == 0:
        fig, (ax1, ax2) = plt.subplots(1,2, figsize=(6,12))
        ax1.imshow(band1)
        ax2.imshow(band2)
    else:
        raise ValueError("Invalid band value.")
    object_type = "Ship" if row.is_iceberg == 0 else "Iceberg"
    plt.title(object_type+";  Angle:  "+str(row.inc_angle))
    plt.show()