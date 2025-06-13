import joblib
import json
import numpy as np
import os
import pandas as pd
import sys

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torch.utils.data.dataloader import default_collate


cwd = os.path.dirname(os.getcwd())
mother_dir = os.path.dirname(cwd) + os.sep
print(mother_dir)
# sys.path.append(os.path.abspath(os.path.join(mother_dir , "MAST_tools")))
sys.path.append(mother_dir)
sys.path.append(cwd)
sys.path.append(os.path.join( os.path.dirname(cwd) ) )

#================================================================
            ##########    TRANSFORMERS   ##########
#================================================================

import torch
import pandas as pd
class ForwardFillImputerTransform:
    def __call__(self, dict):
        """
        Input: torch dict with key 'time' and key 'values'
        Returns: torch dict with key 'time' and key 'values with NaNs forward-filled
        """
        # print('\nCECILE TRANSFORM')
        time = dict['time']
        values = dict['values']
        df = pd.DataFrame(values.T)
        # print('\nBefore forward filling: ', list(df.isna().sum(axis=0)))
        df = df.ffill(axis=0)
        # print('After forward filling: ', list(df.isna().sum(axis=0)))
        return {'time': time,
                'values': df.values.T}

class FillWithZerosImputerTransform:
    def __call__(self, dict):
        """
        Input: torch dict with key 'time' and key 'values'
        Returns: torch dict with key 'time' and key 'values with NaNs filled with zeros
        """
        time = dict['time']
        values = dict['values']
        df = pd.DataFrame(values.T)
        # print('\nBefore filling with zeros: ', list(df.isna().sum(axis=0)))
        df = df.fillna(value=0)
        # print('After filling with zeros: ', list(df.isna().sum(axis=0)))
        
        return {'time': time,
                'values': df.values.T}

from collections import defaultdict

class SamplingtoReferenceTimeTransform:
    def __init__(self, ref_freq):
        self.ref_freq = ref_freq
    def __call__(self, dict):
        """
        Input:
        torch dict with key 'time' and key 'values'
        Returns: torch dict with key 'time' and key 'values with NaNs forward-filled
        """
        time = dict['time']
        values = dict['values']
        # print('\nAfter time rescaling', values.shape[-1] )
        ref_time = []
        for k in range (0, int( (time[-1]-time[0])/(self.ref_freq))+2):
            ref_time. append ( time[0] + k*self.ref_freq )
        # Upsample or subsample by picking the closest value in time to the ref_time
        idx = np.searchsorted(time, ref_time) # reformat to be on the same scale as ref_time
        idx = np. clip(idx, 1, len(time) - 1)
        values_rescaled = values[:,idx]
        # print('\nAfter time rescaling', values_rescaled.shape[-1] )
        return {'time': ref_time,
                'values': values_rescaled}

class SamplewiseNormalizeTransform:
    def __call__(self, d):
        """
        Normalize each sample individually: subtract mean, divide by std.
        Input: dict with 'time' and 'values' [features, time]
        Output: same dict, with values normalized per feature
        """
        time = d['time']
        values = d['values']
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
        if values is not None:
            mean = np.nanmean(values, axis=1, keepdims=True)
            std = np.nanstd(values, axis=1, keepdims=True)
            std[std == 0] = 1.0  # avoid division by zero
            values = (values - mean) / std
        return {'time': time, 'values': values}
        
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


#================================================================
        ##########    END OF TRANSFORMERS   ##########
#================================================================