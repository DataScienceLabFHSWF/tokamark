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
    def __init__(self, parameters_cnn):
        self.param_cnn_x_list = parameters_cnn['x']
        self.param_cnn_y_list = parameters_cnn['y']
        self.dt = parameters_cnn['dt']
    def __call__(self, shot):
        """
        Input:
        torch dict with key 'time' and key 'values'
        Returns: torch dict with key 'time' and key 'values with NaNs forward-filled
        """
        # print('CNN-specific formatting')

        # get min time for all variables of shot
        min_common_time = min(data['values'].shape[-1] for var, data in shot.items())
        # print('min_common_time is ', min_common_time)
        
        # save x data the proper way reduce data to have same time           
        processed_x = {}
        for var, decalage in self.param_cnn_x_list.items():
            # print('shot keys are ', shot.keys())
            # print('var is ', var)
            # print('decalage is ', decalage)
            data = shot[var]
            if decalage=='t':
                processed_x[var] = data['values'][..., :min_common_time-self.dt] 
            elif decalage=='t+dt':
                processed_x[var] = data['values'][..., self.dt:min_common_time] 
            else:
                print('ERROR in decalage')
            # print(f'input shapes for {var}: ', processed_x[var].shape)
        
        # save y data the proper way reduce data to have same time           
        processed_y = {}
        for var, decalage in self.param_cnn_y_list.items():
            # print(var)
            data = shot[var]
            if decalage=='t':
                processed_y[var] = data['values'][..., :min_common_time-self.dt] 
            elif decalage=='t+dt':
                processed_y[var] = data['values'][..., self.dt:min_common_time] 
            else:
                print('ERROR in decalage')
            # print(f'input shapes for {var}: ', processed_y[var].shape)        
        
        # reduce data to have same time 
        array_x = [data.T for var, data in processed_x.items()] 
        array_y = [data.T for var, data in processed_y.items()] 
        # print('input shapes: ', [arr.shape for arr in array_x])
        # print('output shapes: ', [arr.shape for arr in array_y])
        # group arrays by shape for CNN branch
        reshaped_x = self._group_arrays_by_shape(array_x)
        reshaped_x = [arr.squeeze(-1) if arr.shape[-1] == 1 else arr for arr in reshaped_x]
        # print([arr.shape for arr in reshaped_x])
        reshaped_y = self._group_arrays_by_shape(array_y)
        reshaped_y = [arr.squeeze(-1) if arr.shape[-1] == 1 else arr for arr in reshaped_y]
        # print([arr.shape for arr in reshaped_y])
        # get list of windows for x and y
        list_reshaped_x = [ [channel for channel in window] 
                           for window in [ [arr[timestamp] 
                                            for arr in reshaped_x] 
                                            for timestamp in range(min_common_time-self.dt) ] ]
        # print('\nlen(list_reshaped_x)', len(list_reshaped_x))
        list_reshaped_y = [ window[0]
                           for window in [ [arr[timestamp]
                                            for arr in reshaped_y] 
                                            for timestamp in range(min_common_time-self.dt) ] ]
        # print('len(list_reshaped_y)', len(list_reshaped_y))

        return list_reshaped_x, list_reshaped_y

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
            # print(transposed.shape)
            reshaped_list.append(transposed)
        # print([ arr.shape for arr in reshaped_list ])
        return reshaped_list