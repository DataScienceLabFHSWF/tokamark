from collections import defaultdict
import json
import joblib
import numpy as np
import os
import pandas as pd
import sys
import torch
from torch.utils.data.dataloader import default_collate
from typing import Optional

cwd = os.path.dirname(os.path.abspath(__file__))
mother_dir = os.path.dirname(cwd) + os.sep
sys.path.append(os.path.abspath(os.path.join(mother_dir , "MAST_tools")))
sys.path.append(mother_dir)

from signal_segmentation import (segment_data_in_time_windows,
                                 segment_sample)


from signal_utils import MASTSignalManager  
from store_utils import MASTStorageManager


class SegmenterTransform(object):
    
    def __init__(self, time_window_sec, time_step, offset):
        self.time_intervals_checks= False
        if not (
                time_window_sec > 0 and \
                time_step > 0 and \
                offset >= 0 and \
                time_step < time_window_sec 
            ):
            self.time_intervals_checks = True
            raise ValueError(
                "Invalid parameters for time window, time step or offset. " +
                "Ensure that time_window_sec > 0, time_step > 0, offset >= 0, " +
                "time_step < time_window_sec and offset <= time_window_sec."
            )
        
        self.time_window_sec = time_window_sec
        self.time_step =  time_step
        self.offset =  offset

    def __call__(self, batch):    
        all_x_segments = []
        all_y_segments = []

        for sample in batch:
            if sample is None:
                continue

            list_x, list_y = segment_sample(
                sample, 
                self.time_window_sec, 
                self.time_step, 
                self.offset
                )

            if list_x is None or list_y is None:
                print("Warning: problem with data lengths after segmentation: list_x or list_y is None")
                continue
            
            if len(list_x) != len(list_y):
                print("Warning: problem with data lengths after segmentation: lengths of list_x and list_y differ")
                print(f"Lengths of lists: len(list_y) = {len(list_y)}, len(list_x) = {len(list_x)}")
                continue

            if len(list_y) == 0 or len(list_x) == 0:
                print("Warning: problem with data lengths after segmentation")
                print(f"Lengths of lists: len(list_y) = {len(list_y)}, len(list_x) = {len(list_x)}")
                continue
                        
            for x_segment, y_segment in zip(list_x, list_y):
                # Extract x values
                x_values = [
                    signal_dict["values"]
                    for signal_dict in x_segment["source_name-signal_name"]
                ]
                # shape: [num_signals, nr_features, time_window_length]
                
                # Extract y values
                y_values = [
                    signal_dict["values"]
                    for signal_dict in y_segment["source_name-signal_name"]
                ]
                # shape: [num_signals, nr_features, time_window_length]
                
                all_x_segments.append(x_values) # shape: [list_x_length, num_signals, nr_features, time_window_length]
                all_y_segments.append(y_values) # shape: [list_y_length, num_signals, nr_features, time_window_length]
            
        if not all_x_segments or not all_y_segments:
            return None  # or return empty batch dicts
        
        return {'x': all_x_segments, 'y': all_y_segments}
    
class PCATransform(object):
    """Use a pre-fitted PCA function to transform input data.

    Parameters
    ----------
    pca_model_path : str
        Path to fitted pca model joblib file.
    """

    def __init__(self, models):
        self.models = models

    def __call__(self, sample):
        vals, time, source_signal = sample
        
        if vals is None or len(vals) == 0:
            return None

        # Select model type and signal from those available in the dictionary.
        pca_model =  self.models["pca"][source_signal]
                                
        # Apply Scaler and PCA
        if not np.isnan(vals).any():
            
            #Import models
            pca = pca_model["pca"]
            scaler = pca_model["scaler"]
            
            # Transoform data in PCA space
            x_scaled = scaler.transform(vals.T) 
            x_transform = pca.transform(x_scaled)
            #vals = scaler.inverse_transform(x_transform).T
            vals = x_transform.T
           
            
            return (vals, time, source_signal)
        else:
            return None

class ImputerTransform(object):
    """Use a pre-fitted mean imputer to transform input data.

    Parameters
    ----------
    model: [dict]
        Dictionaries containing the joblib models for pca and imputer.
    global_imputer_path : [str]
        Path to the file containing the global averaged imputed values.
    """

    def __init__(self, models, global_imputer_path):
        self.models = models
        self.global_imputer_path = global_imputer_path
          
        with open(global_imputer_path, 'r') as f:
            data = json.load(f)
        self.data = data

    def __call__(self, sample):
        vals, time, source_signal = sample
        
        # Signal was empty 
        if vals.size == 0:
            source_name, signal_name = source_signal.split('-')
            try:
                vals = self.data.get("data", [])[signal_name]
                vals = np.repeat(vals[:, np.newaxis], len(time), axis=1)
            except:
                return None

        # Apply mean imputer if signal is not empty and any nan are found
        if vals.size>0 and np.isnan(vals).any():
            try:
                mean_imputer = self.models["imputer"][source_signal]
                x_transformed = mean_imputer.transform(vals.T)
                vals = x_transformed.T
            except ValueError as e:
                print(f"ValueError {e}")
                return None

        return (vals, time, source_signal)
      
class ComposeTransform(object):
    """Compose transforms and apply them in series checking for None return values

    Parameters
    ----------
    transforms : list[callable[tuple]]
        List containing the names of the transforms
    """
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, sample):
        for transform in self.transforms:
            if sample is None:
                return None
            sample = transform(sample)
        return sample
