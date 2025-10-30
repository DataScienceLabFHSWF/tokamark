from typing import List, Dict, Any
import numpy as np


# ======================================================================================================================
class WindowTruncationTransform:

    # ------------------------------------------------------------------------------------------------------------------
    def __init__(
        self,
        x_timestamp, 
        y_timestamp
    ):

        self.x_timestamp = x_timestamp
        print('x_timestamp', x_timestamp)
        print('y_timestamp', y_timestamp)
        self.y_timestamp = y_timestamp

    # ------------------------------------------------------------------------------------------------------------------
    def __call__(self, shot):
        new_shot = []

        for window in shot:
            new_window = {}

            dict_x = window['x']
            dict_y = window['y']

            for var in dict_x.keys():
                # print(var)
                dict_x[var]['time'] = dict_x[var]['time'][:self.x_timestamp]
                dict_x[var]['values'] = dict_x[var]['values'][:,:self.x_timestamp,...]
                # print(dict_x[var]['time'].shape)
            new_window['x'] = dict_x

            for var in dict_y.keys():
                dict_y[var]['time'] = dict_y[var]['time'][:self.y_timestamp]
                dict_y[var]['values'] = dict_y[var]['values'][:,:self.y_timestamp,...]
                # print(dict_y[var]['time'].shape)
            new_window['y'] = dict_y

            if window['exog']:
                dict_exog = window['exog']
                

            new_shot.append(new_window)

        return new_shot

    # ------------------------------------------------------------------------------------------------------------------
