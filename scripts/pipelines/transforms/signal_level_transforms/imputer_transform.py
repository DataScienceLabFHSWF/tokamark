import json
import joblib
import numpy as np
import os
import sys

cwd = os.path.dirname(os.path.abspath(__file__))
mother_dir = os.path.dirname(cwd) + os.sep
sys.path.append(mother_dir)

from .segmenter_transform import (segment_data_in_time_windows,
                                 segment_sample)


from MAST_tools.signal_utils import MASTSignalManager  
from MAST_tools.store_utils import MASTStorageManager


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
        try:
            vals, time, source_signal = sample["values"], sample["time"], sample["source-signal"]
        except KeyError as e:
            print(f"KeyError: {e}. Sample is missing required keys.")
            return None
        
        if vals is None or len(vals) == 0:
            return None
        if time is None or len(time) == 0:
            return None
        
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

        return {"values":vals, "time":time, "source-signal":source_signal}
      