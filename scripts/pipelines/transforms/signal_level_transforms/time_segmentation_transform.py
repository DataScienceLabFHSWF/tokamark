
from collections import defaultdict
import torch

class SegmenterTransform(object):
    
    def __init__(
        self,  
        time_window_sec, 
        time_step, 
        offset
        ):
        
        
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

    def __call__(self, shot):    
        return segment_shot(
                shot,  
                self.time_window_sec, 
                self.time_step, 
                self.offset
            )

    
def segment_data_in_time_windows(
    values,
    times,
    time_window, 
    time_step,
    offset=0.0
    ):
    """ Segment the "values" and "times" into smaller segments 
        each containing the original data in a given time window.
    Parameters
    ----------
    values : np.ndarray
        The values to segment, shape: (n_features, time_steps).
    times : np.ndarray
        The times corresponding to the values, shape: (time_steps,).
    time_window : float
        The length of the time window in seconds to segment the x and y values.
    time_step : float
        The step in seconds to move the time window.
    offset : float, optional
        The offset in seconds to start the time window from the end of the signal, by default 0.0.
        This is also used to offset the y segments in time with respect to the x segments.
    Returns
    -------
        Two lists one for "values" and one for "times" containing segments.
    """
    
    assert time_window >= time_step, \
        "Time window must be greater than time step."   

    values = torch.tensor(values,dtype=torch.float32)  # shape: (n_features, time_steps)
    times = torch.from_numpy(times)    # shape: (time_steps,)

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

    return segments_v, segments_t


def create_map(shot):
    """Create indexed map that contains mini-dictionaries, 
        one per segment of values and times in the shot"""
    # Create a defaultdict to hold the segments
    data_map = defaultdict(list)

    # For each signal in the base dictionary
    for source_signal_names, sample in shot.items():
        try:
            values = sample["values"] # list of segments
            times = sample["time"] # list of segments
        except KeyError as e:
            print(f"KeyError: {e}. Sample is missing required keys.")
            return None
        
        
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
                "signal_name": source_signal_names, 
                "values": values_segment, 
                "time": time_segment
            })
            
    return data_map

def segment_shot(
    shot,  
    time_window_sec, 
    time_step, 
    offset):
    """Segment the dictionaries contained in sample into smaller 
       dictionaries each containing a segment of the original values.
       
       These segments are created by sliding a time window of length
       time_window_sec over the x and y values with a step of time_step. This algorithm is
       implemented in the function segment_data_in_time_windows.
       
       HINT: for forecasting purposes, the segments of y are 
       forward in time with respect to the segments of x. 
    

    Parameters
    ----------
    shot : dict
        An item from the MastDataset single shot level transform
    time_window_sec : int
        The length of the time window in seconds to segment the x and y values.
    time_step : int
        The step in seconds to move the time window.
    offset : int
        The offset in seconds to start the time window from the end of the signal.
    Returns
    -------
        A list of sub-x dictionaries each one containing a time segment 
        of the original x, for all the signals in the original x. 
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
        of the original x.
    """
    if not shot:
        return None
    if type(shot) is not dict:
        raise TypeError(f"---------------------- Shot must be a dictionary and not {type(shot)}.")
   
    for _, sample in shot.items():
        try:
            values, times = sample["values"], sample["time"]
        except KeyError as e:
            print(f"KeyError: {e}. Sample is missing required keys.")
            return None

        # Segmet "values" and "times" in x and y
        try:
            segments_v, segments_t = segment_data_in_time_windows(
                values,
                times, 
                time_window_sec, 
                time_step, 
                offset
            )
            sample["values"] = segments_v
            sample["time"] = segments_t
        except AssertionError as e:
            print(f"AssertionError: {e}")
            return None
        
    # Create map of segments
    x_map = create_map(shot)
              
    # Create a list of new mini dictionaries x, one for each temporal segment
    x_list = []
    for idx in range(len(x_map)):
        x_mini_dict = {
            "sources_signals": x_map[idx]
        }
        x_list.append(x_mini_dict)
        
    return x_list


if __name__ == "__main__":
    
    train_shots = [30207]
    data_names =["magnetics-flux_loop_flux"]
    target_names =["magnetics-b_field_tor_probe_saddle_voltage"]

    import os
    import sys
    
    sys.path.append("./scripts/MAST_tools")
    from MAST_dataset import MastDataset_test as MastDataset
  
    shot = MastDataset(
        local=False,
        shots_list=train_shots,
        source_signal_list=data_names + target_names,
        signal_level_transform_map=None,
        shot_level_transform=None
    )
    
    time_window_length =  0.01
    step = 0.005
    offset = 0.003
    
    time_settings = {
        "time_window_length": time_window_length,
        "step":step,
    }

    # Working
    list_x, list_y = test_2(
        shot[0], 
        data_names, 
        target_names, 
        time_window_length, 
        step, 
        offset
        )
    
    # Printing the first two segments of x and y
    x0=list_x[-1]
    y0=list_y[-1]
    
    x1=list_x[-2]
    y1=list_y[-2]
    
    print(f"x0 times: {x0['source_name-signal_name'][0]['time']}")
    print(f"x1 times: {x1['source_name-signal_name'][0]['time']}")
    print(f"y0 times: {y0['source_name-signal_name'][0]['time']}")
    print(f"y1 times: {y1['source_name-signal_name'][0]['time']}")