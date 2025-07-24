from collections import defaultdict
import numpy as np
import os
import pandas as pd
import sys
from typing import Optional

cwd = os.path.dirname(os.path.abspath(__file__))
mother_dir = os.path.dirname(cwd) + os.sep
sys.path.append(os.path.abspath(os.path.join(mother_dir , "MAST_tools")))
sys.path.append(mother_dir)


from signal_utils import MASTSignalManager  
from store_utils import MASTStorageManager


def segment_data_in_time_windows(
    base_dataset,
    time_window, 
    time_step,
    offset=0.0
    ):
    """Segment the "values" and "times" Torch tensors contained in a base_dataset.
       into smaller tensors each containing a segment of the original data in a
       given time window.


    Parameters
    ----------
    base_dataset : tuple
        A sample (x,y) where both x and y are dictionaries (or None) returned by __getitem__.
    time_window : float
        The length of the time window in seconds to segment the x and y values.
    time_step : float
        The step in seconds to move the time window.
    offset : float, optional
        The offset in seconds to start the time window from the end of the signal, by default 0.0.
        This is also used to offset the y segments in time with respect to the x segments.
    Returns
    -------
        Tuple (x, y) where their "values" and "times" have been segmented.
    """
    
    assert time_window >= time_step, \
        "Time window must be greater than time step."   
        
    x, y = base_dataset

    def split_in_time_segments(item, time_window, time_step, offset):
        for signal in item["source_name-signal_name"]:
            values = signal["values"]  # shape: (n_features, time_steps)
            times = signal["time"]     # shape: (time_steps,)

            start_time = times[-1].item() - time_window - offset
            segments_v = []
            segments_t = []

            while start_time >= times[0].item():
                        
                # Find the indeces where the times is in the time window
                mask = (times >= start_time) & (times < start_time + time_window)
                        
                # Return list of True indeces
                idx = mask.nonzero(as_tuple=True)[0]
                
                # If no end index is found, take the whole time series
                if len(idx) == 0:
                    segments_v.insert(0,values)
                    segments_t.insert(0,times)
                    break

                val_segment =  values[:, idx[0]:idx[-1] + 1]
                time_segment = times[idx[0]:idx[-1] + 1]

                segments_v.insert(0,val_segment)
                segments_t.insert(0,time_segment)

                start_time -=  time_step

            signal["values"] = segments_v
            signal["time"] = segments_t

        return item
        
    x = split_in_time_segments(x, time_window, time_step, offset)
    y = split_in_time_segments(y, offset, time_step, offset=0.0)
    
    return x, y

def segment_sample(sample, time_window_sec, time_step, offset):
    """Segment the x,y dictionaries contained in sample into smaller 
       dictionaries each containing a segment of the original x, y values.
       
       These segments are created by sliding a time window of length
       time_window_sec over the x and y values with a step of time_step. This algorithm is
       implemented in the function segment_data_in_time_windows.
       
       HINT: for forecasting purposes, the segments of y are 
       forward in time with respect to the segments of x. 
    

    Parameters
    ----------
    sample : tuple
        A sample (x,y) where both x and y are dictionaries (or None) returned by __getitem__
    time_window_sec : int
        The length of the time window in seconds to segment the x and y values.
    time_step : int
        The step in seconds to move the time window.
    offset : int
        The offset in seconds to start the time window from the end of the signal.
    Returns
    -------
        A list of sub-x and sub-y dictionaries each one containing a time segment 
        of the original x and y values, for all the signals in the original x and y. 
        For instance, the first element of the list will contain:
        {       
                "shot_id": x["shot_id"]
                "source_name-signal_name": [
                       {
                           "signal_name": signal["names"], 
                            "values": values_segment, 
                            "time": time_segment
                        },
                        {
                          ...
                        },
                        ...
                    ]
        }
                
        all structures { ... } correspond to the first time segment
        of the original x and y values.
    """
    if sample is None:
        return None, None

    # Segmet "values" and "times" in x and y
    try:
        x,y = segment_data_in_time_windows(
            sample, 
            time_window_sec, 
            time_step, 
            offset
        )
    except AssertionError as e:
        print(f"AssertionError: {e}")
        return None, None
    
    # Create indexed map that contains mini-dictionaries, one per segment in
    # values and times
    def create_map(data):
        data_map = defaultdict(list)
    
        # For each signal in the base dictionary
        for signal in data["source_name-signal_name"]:
            values = signal["values"] # list of segments
            times = signal["time"] # list of segments
            
            # segment_data_in_time_windows return a list of time segments of fizxed length
            # However, the last segment may be shorter than the others is the signal is not long enough
            # to contain a full time window.
            # We will use the first segment length to determine the length of the segments and compare to last segment.
            # Since this could be true for one signal only we cannot get rid of one segment for it and keep 
            # the same number of segments for all the other signals. This will not work when batching minibatches.
            # Therefore, we will always remove the last segment independentkly of its length.
            for idx, (values_segment, time_segment) in enumerate(zip(values, times)):
                if idx == len(times) - 1:
                    continue    # Skip last segment, since this might be shorter than the others, see explanation above
                
                data_map[idx].append({
                    "signal_name": signal["names"], 
                    "values": values_segment, 
                    "time": time_segment
                })
                
        return data_map

    x_map = create_map(x)
    y_map = create_map(y)
              

    # Create a list of new mini dictionaries x and y, one for each temporal segment
    x_list = []
    y_list = []
    for idx in range(len(x_map)):
        x_mini_dict = {
            "shot_id": x["shot_id"],
            "source_name-signal_name": x_map[idx]
        }
        x_list.append(x_mini_dict)
        
    for idy in range(len(y_map)):
        y_mini_dict = {
            "shot_id": y["shot_id"],
            "source_name-signal_name": y_map[idy]
        }
        y_list.append(y_mini_dict)
    
    #crop y_list from the beginning so that its length matches x_list, you can do
    if len(y_list) > len(x_list):
        y_list = y_list[-len(x_list):]
    if len(x_list) > len(y_list):
        x_list = x_list[-len(y_list)]
        
    return x_list, y_list


def roll_time_window_backward(
    time_window_length: float,
    start: float, 
    step: float, 
    stop: float):
    """Rolls a time window backward from a start time to a stop time with a given step.

    Parameters
    ----------
    time_window_length : float
        The length of the time window in seconds.
    start : float
        The start time in seconds from which to roll the time window backward.
    step : float
        The step in seconds to roll the time window backward.
    stop : float
        The stop time in seconds until which to roll the time window backward.

    Returns
    -------
        list[list]
        A list of time windows, each represented as a list of two floats [start, stop].
        Each time window is of length `time_window_length` and rolled backward by `step`
        seconds until the `stop` time is reached.
    """
    
    assert stop < start and step > 0, \
        "stop < start and step > 0 not verified"
    
    time_windows = []
    t = start
    while t >= stop:
        time_windows.append([t,t-time_window_length])
        t -= step
    return time_windows

def adjust_time_windows(time_windows:list[list], offest:float):
        new_time_windows = []
        for time_interval in time_windows:
            new_time_windows.append([time_interval[1],time_interval[1]+offset])
        return new_time_windows
    

def test_2(shots, time_window_length, step, offset):
    list_x, list_y = segment_sample(
        shots[0], 
        time_window_length,
        step,
        offset
    )
    return list_x, list_y


if __name__ == "__main__":
    
    train_shots = [30207]
    data_names =["magnetics-flux_loop_flux"]
    target_names =["magnetics-b_field_tor_probe_saddle_voltage"]

    from MAST_pytorch_training import MASTDataset, PCATransform,  ImputerTransform
    
    shots = MASTDataset(
        False,
        train_shots, 
        data_names=data_names, 
        target_names=target_names,
        transform_data = None,
        transform_target = None
        )
    
    time_window_length =  0.01
    step = 0.005
    offset = 0.003
    
    time_settings = {
        "time_window_length": time_window_length,
        "step":step,
    }

    # Working
    list_x, list_y = test_2(shots, time_window_length, step, offset)
    
    # Printing the first two segments of x and y
    x0=list_x[-1]
    y0=list_y[-1]
    
    x1=list_x[-2]
    y1=list_y[-2]
    
    print(f"x0 times: {x0['source_name-signal_name'][0]['time']}")
    print(f"x1 times: {x1['source_name-signal_name'][0]['time']}")
    print(f"y0 times: {y0['source_name-signal_name'][0]['time']}")
    print(f"y1 times: {y1['source_name-signal_name'][0]['time']}")