import numpy as np
import os
import sys

import torch


cwd = os.path.dirname(os.getcwd())
mother_dir = os.path.dirname(cwd) + os.sep
sys.path.append(os.path.abspath(os.path.join(mother_dir , "MAST_tools")))
sys.path.append(mother_dir)
sys.path.append(cwd)
sys.path.append(os.path.join( os.path.dirname(cwd) ) )

from collections import defaultdict


class CNNSpecificTransform:

    def __call__(self, list_samples):
        """
        Input:
        torch dict with key 'source-signal' containing dict with keys 'time' and key 'values'
        Returns: 
        listtorch dict with key 'time' and key 'values with NaNs forward-filled
        """
        # print('CNN-specific formatting')
        
        cnn_samples = []

        for sample in list_samples:
            # reduce data to have same time 
            array_x = [data['values'].T for var, data in sample['x'].items()]
            array_y = [data['values'].T for var, data in sample['y'].items()]

            # group arrays by shape for CNN branch
            reshaped_x = self._group_arrays_by_shape(array_x)
            reshaped_x = [arr.squeeze(0) for arr in reshaped_x]
            reshaped_x = [arr.squeeze(-1) if arr.shape[-1] == 1 else arr for arr in reshaped_x]
            reshaped_y = self._group_arrays_by_shape(array_y)
            reshaped_y = [arr.squeeze(0) for arr in reshaped_y]
            reshaped_y = [arr.squeeze(-1) if arr.shape[-1] == 1 else arr for arr in reshaped_y]

            cnn_samples.append((reshaped_x, reshaped_y))
        
        return cnn_samples

    def _group_arrays_by_shape(self, arrays):
        grouped = defaultdict(list)
        for arr in arrays:
            grouped[arr.shape].append(arr)
        reshaped_list = []
        for same_shape_group in grouped.values():
            stacked = np.stack(
                        [a for a in same_shape_group]
                    )
            transposed = np.transpose(stacked, axes=[1, 0] + list(range(2, stacked.ndim)))
            reshaped_list.append(transposed)
        return reshaped_list