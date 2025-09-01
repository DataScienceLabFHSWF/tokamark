import json
import joblib
import numpy as np
import os
import sys

cwd = os.path.dirname(os.path.abspath(__file__))
mother_dir = os.path.dirname(cwd) + os.sep
sys.path.append(mother_dir)


class ImputerTransform(object):
    """Use a pre-fitted mean imputer to transform input data.

    Parameters
    ----------
    model: [dict]
        Dictionaries containing the joblib models for pca and imputer.
    global_imputer_path : [str]
        Path to the file containing the global averaged imputed values.
    """

    def __init__(self, model_imputer, global_imputer_path):
        self.model_imputer = model_imputer
        self.global_imputer_path = global_imputer_path
          
        with open(global_imputer_path, 'r') as f:
            data = json.load(f)
        self.data = data

    def __call__(self, sample):
        try:
            vals, time = sample["values"], sample["time"]
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
                x_transformed = self.model_imputer.transform(vals.T)
                vals = x_transformed.T
            except ValueError as e:
                print(f"ValueError {e}")
                return None

        return {"values":vals, "time":time}
      