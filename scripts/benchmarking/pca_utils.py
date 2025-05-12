
import cv2
import fsspec
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import os
import pandas as pd
import s3fs
import xarray as xr
import zarr
import json

import sys
cwd = os.path.dirname(os.path.abspath(__file__))
mother_dir = os.path.dirname(cwd) + os.sep
mast_tools_path = os.path.abspath(os.path.join(mother_dir , "MAST_tools"))
sys.path.append(mast_tools_path)
from signal_utils import MASTSignalManager  
from store_utils import MASTStorageManager

def read_signals(filepath: str)->dict[str:int]:
    """
    filepath: path to file containing all 
    signal names and their multeplicity (nr. of traces)

    Output: Dictionary with signal names as keys and 
    number of traces as values
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

def signals_average_across_shots(store_manager : MASTStorageManager, 
                                 filepath:str,
                                 local : bool,
                                 singularity: bool):
    """
    Summary: calucalte average values of MAST signals across all shots available. 
    Signals are listed inside the file found at filepath.
    """

    # Get all shot ids
    shot_ids = store_manager.list_all_shots(local=local, singularity = singularity)

    # Read file containing signal names and the number of channels for each signal
    data_set = read_signals(filepath)

    data_set_counter = {}
    data_set_sums = {}

    # Loop through shots
    for shot_id in shot_ids[:2]:
        store = store_manager.make_shot_store(shot_id, local=local, singularity = singularity)

        for key, value in data_set.items():
            
            # Retrieve signal group and name
            sub_keys = key.split("/")
            group = sub_keys[0]
            signal_name = sub_keys[1]

            # Form instance of SIGNAL
            sig = MASTSignalManager(group, signal_name, shot_id)

            # Initialize datasets that will run through shots
            if value not in data_set_counter:
                data_set_counter[signal_name] = 0
            if value not in data_set_sums:
                data_set_sums[signal_name] = None

            try:
                vals = sig.get_values(store)
                data_set_counter[signal_name] += 1

                if data_set_sums[signal_name] is None:
                    # Initialize the summed array with the first array's shape and data type
                    data_set_sums[signal_name] =  np.nan_to_num(vals)
                else:
                    data_set_sums[signal_name] += np.nan_to_num(vals)

            except Exception as e:
                print(f"Exception: {e}")

    # Calculate average 
    averaged_arrays = {}
    data = []
    for value, summed_array in data_set_sums.items():

        counter = data_set_counter[value]
        if summed_array is not None and counter > 0:
            averaged_arrays[value] = summed_array / counter
            data.append({value: averaged_arrays[value].tolist()})
        else:
            print(f"No valid data to average for value '{value}'.")
    
    json_data = {"data":data}
    with open("averaged_arrays.json", "w") as f:
        json.dump(json_data, f, indent=4)



if __name__ == "__main__":
    breakpoint()
    store_manager = MASTStorageManager(local_root_path = "/srv")
    signals_average_across_shots(store_manager,"data/list_of_signals.txt", False, True)