"""
Utilities for processing MAST signals.

How to use it: python pca_utils.py
"""

import numpy as np
import os
import json

import sys
cwd = os.path.dirname(os.path.abspath(__file__))
mother_dir = os.path.dirname(cwd) + os.sep
mast_tools_path = os.path.abspath(os.path.join(mother_dir , "MAST_tools"))
sys.path.append(mast_tools_path)
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
    

def signals_average_across_shots(store_manager : MASTStorageManager, 
                                 shot_ids : list[int],
                                 filepath:str,
                                 local : bool,
                                 singularity: bool):
    """
    For each shot:

    1- considers all signal vectors listed in the 
    data/list_of_signals.txt 

    2- calculates the average value of each channel 
    in the signal. 

    3- Set to zero channels that contain NaN;

    3- returns the average value across shots in a json format:
    {
        "data": [
            {
                "signal_name": [ 
                    [avergae channel 0 ],
                    [average channel 1],
                    ...
                ],
                ...
            }
        ]
    }
    """

    # Read file containing signal names and the number of channels for each signal
    data_set = read_signals(filepath)

    data_set_counter = {}
    data_set_sums = {}

    # Loop through shots
    for nr,shot_id in enumerate(shot_ids[:5]):

        if nr>=0 :
            print(f"Shot number {nr+1} out of {len(shot_ids)}")
        store = store_manager.make_shot_store(shot_id, local=local, singularity = singularity)

        for key, value in data_set.items():
            # Retrieve signal group and name
            group, signal_name = key.split("/")

            # Form instance of SIGNAL
            sig = MASTSignalManager(group, signal_name, shot_id)

            # Prepare tracking
            data_set_counter.setdefault(signal_name, np.zeros(value))
            data_set_sums.setdefault(signal_name, None)

            try:
                vals = sig.get_values(store)

                mean_vals = np.mean(vals, axis=1) # Returns NaN if NaN is in vals

                # Check mean_vals are all numeric different from NaN and Inf
                for i, mean_val in enumerate(mean_vals):

                    if is_finite_numeric_array(mean_val):
                        data_set_counter[signal_name][i] += 1
                    else:
                        mean_vals[i]=0

                # Add mean_vals to data_set_sums
                if data_set_sums[signal_name] is None:
                    data_set_sums[signal_name] =  mean_vals
                else:
                    data_set_sums[signal_name] += mean_vals

            except Exception as e:
                print(f"Exception: {e}")

    # Calculate average 
    data = []
    for signal_name, summed_array in data_set_sums.items():
        
        counter = data_set_counter[signal_name]
        
        averaged_channels = []
        for channel, count in zip(summed_array, counter):
            if channel and count> 0:
                averaged_channels.append(channel / count)
            else:
                print(f"No valid data to average for value '{signal_name}'.")

        data.append({signal_name: averaged_channels})
    
    json_data = {"data":data}
    with open("averaged_arrays.json", "w") as f:
        json.dump(json_data, f, indent=4)



if __name__ == "__main__":
     # Get all shot ids
    local = False
    singularity = True
    store_manager = MASTStorageManager(local_root_path = "/srv")
    shot_ids = store_manager.list_all_shots(local= local, singularity = singularity)
    signals_average_across_shots(store_manager,
                                shot_ids[:5], 
                                "data/list_of_signals.txt", 
                                local,
                                singularity)