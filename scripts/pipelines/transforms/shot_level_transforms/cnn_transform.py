import numpy as np
from collections import defaultdict
# ======================================================================================================================
class CNNTransform:

    def __init__(self):
        # dictionary that persists across calls
        self.var_groups = {"x_past": None, 
                           "y_past": None, 
                           "x_future": None, 
                           "y_future": None}

    # ------------------------------------------------------------------------------------------------------------------
    def __call__(self, window):

        # x_past
        array_x_past = [ data['values'][...,-1:].T for var, data in window['x_past'].items() ] 
        names_x_past = list(window['x_past'].keys())
        # x_future
        array_x_future = [ data['values'][...,:1].T for var, data in window['x_future'].items() ] 
        names_x_future = list(window['x_future'].keys())

        # y_past
        array_y_past = [ data['values'][...,-1:].T for var, data in window['y_past'].items() ] 
        names_y_past = list(window['y_past'].keys())
        # y_future
        array_y_future = [ data['values'][...,:1].T for var, data in window['y_future'].items() ] 
        names_y_future = list(window['y_future'].keys())

        # x
        array_x = array_x_past + array_x_future
        names_x = names_x_past + names_x_future

        # y 
        array_y = array_y_past + array_y_future
        names_y = names_y_past + names_y_future
        
        return {'window_index': window['window_index'],
                'x': array_x,
                'y': array_y}


