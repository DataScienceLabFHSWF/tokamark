
import cv2
import fsspec
from MAST_signal import SIGNAL
import MAST_store as MASTbucket
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import os
import pandas as pd
import s3fs
import xarray as xr
import zarr
import json



def read_signals(filename: str)->dict[str:int]:
    """
    Input: filename path to file to read containing all 
    signal names and their multeplicity (nr. of traces)

    Output: Dictionary with signal names as keys and 
    number of traces as values
    """

    signals = dict()
    try:
        with open(filename, "r") as file:
            for line in file:
                signals[line.split()[0]] = int(line.split()[1])
    except FileNotFoundError:
        print(f"Error: File not found at '{filepath}'")
        return {}
    except Exception as e:
        print(f"An error occurred while reading the file '{filepath}': {e}")
        return {}
    return signals

def test():
    # Get all shot ids
    shot_ids = MASTbucket.list_all_shots()

    # Read file containing signals
    data_set = read_signals("tmp/list_of_signals.txt")

    data_set_counter = {}
    data_set_sums = {}

    # Loop through shots
    for shot_id in shot_ids[:1]:
        store= MASTbucket.make_store(shot_id)

        for key, value in data_set.items():
            
            # Retrieve signal group and name
            sub_keys = key.split("/")
            group = sub_keys[0]
            signal_name = sub_keys[1]

            # Form instance of SIGNAL
            sig = SIGNAL(group, signal_name, shot_id)

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
    test()