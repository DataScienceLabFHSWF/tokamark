import numpy as np
import os
import pandas as pd
import random
import sys

cwd = os.path.dirname(os.path.abspath(__file__))
mother_dir = os.path.dirname(cwd) + os.sep
sys.path.append(os.path.abspath(os.path.join(mother_dir , "MAST_tools")))
sys.path.append(mother_dir)

from signal_utils import MASTSignalManager  
from store_utils import MASTStorageManager

def is_finite_numeric_array(arr):
    # Check if array is numeric
    if not np.issubdtype(arr.dtype, np.number):
        return False
    # Check if all values are finite (no NaN or Inf)
    return np.isfinite(arr).all()


def read_signals(filepath: str)->dict[str:int]:
    """
    filepath: path to file containing all 
    signal names and their multeplicity (nr. of channels)

    Output: Dictionary with signal names as keys and 
    number of traces as values. Empty dictionary if
    exception raised during reading.
    """

    signals = dict()
    try:
        with open(filepath, "r") as file:
            for line in file:
                signals[line.split()[0]] = int(line.split()[1])
    except FileNotFoundError:
        print(f"Error: File not found at '{filepath}'")
        return {}
    except Exception as e:
        print(f"An error occurred while reading the file '{filepath}': {e}")
        return {}
    return signals


def shuffle_shot_ids(shot_ids, N, seed=None):
    random.seed(seed)
    random.shuffle(shot_ids)
    return shot_ids[:N]

def make_dataframe_from_shot_ids(store_manager : MASTSignalManager, 
                    shot_ids: list[int], 
                    group : str, 
                    signal_name: str,
                    local=True,
                    singularity=False):
    """Return a dataFrame from concateneted signals

    Parameters
    ----------
    ...
    local : bool, optional
        get data from local storage
    singularity : bool, optional
        if running within singularity the data storage must be bound to 
        a directory within the singularity. 
    """

    channels = None
    shot_list = []
    for shot_id in shot_ids:
        store = store_manager.make_shot_store(shot_id=shot_id, local=local, singularity=singularity)
        sig = MASTSignalManager(group, signal_name, shot_id)

        if channels == None:
            channels = list(sig.get_channel_names(store))

        df = pd.DataFrame(sig.get_values(store).T, columns=channels)
        shot_list.append(df)
    
    return channels, shot_list

