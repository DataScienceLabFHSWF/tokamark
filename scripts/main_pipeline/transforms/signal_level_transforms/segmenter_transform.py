
from collections import defaultdict
import torch

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
    
    
def segment_data_in_time_windows(
    base_shot,
    data_names,
    target_names,
    time_window, 
    time_step,
    offset=0.0
    ):
    """Segment the "values" and "times" Torch tensors contained in a base_shot.
       into smaller tensors each containing a segment of the original data in a
       given time window.


    Parameters
    ----------
    base_shot : tuple
        A sample returned by __getitem__.
    data_names : list[str]
        List of data names to segment for the data part of the sample.
    target_names : list[str]
        List of data names to segment for the target part of the sample.
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

    def split_in_time_segments(shot, names, time_window, time_step, offset):
        for signal_name, signal_data in shot.items():
            
            source_signal_names = signal_data.get("source-signal")
            if source_signal_names not in names:
                continue
            
            breakpoint()
            values = torch.tensor(signal_data["values"],dtype=torch.float32)  # shape: (n_features, time_steps)
            times = torch.from_numpy(signal_data["time"])    # shape: (time_steps,)

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

            shot[source_signal_names]["values"] = segments_v
            shot[source_signal_names]["time"] = segments_t
    
        return shot
    breakpoint()   
    base_shot = split_in_time_segments(base_shot, data_names, time_window, time_step, offset)
    base_shot = split_in_time_segments(base_shot, target_names, offset, time_step, offset=0.0)
    
    return base_shot

def segment_sample(
    shot, 
    data_names, 
    target_names, 
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
        An item from the MastDataset
    data_names : list[str]
        List of data names to segment for the data part of the sample.
    target_names : list[str]
        List of data names to segment for the target part of the sample.
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
    if shot is None:
        return None, None

    # Segmet "values" and "times" in x and y
    try:
        shot = segment_data_in_time_windows(
            shot, 
            data_names,
            target_names,
            time_window_sec, 
            time_step, 
            offset
        )
    except AssertionError as e:
        print(f"AssertionError: {e}")
        return None, None
    
    # Create indexed map that contains mini-dictionaries, one per segment in
    # values and times
    def create_map(shot, names):
        data_map = defaultdict(list)
    
        # For each signal in the base dictionary
        for signal_name, signal_data in shot.items():
            if signal_data.get("source-signal") not in names:
                continue
            values = signal_data["values"] # list of segments
            times = signa_data["time"] # list of segments
            
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
                    "signal_name": signal_name, 
                    "values": values_segment, 
                    "time": time_segment
                })
                
        return data_map

    x_map = create_map(shot, data_names)
    y_map = create_map(shot, target_names)
              

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

    
def test_2(shot, data_names, target_names, time_window_length, step, offset):
    list_x, list_y = segment_sample(
        shot, 
        data_names, 
        target_names,
        time_window_length,
        step,
        offset
    )
    return list_x, list_y

if __name__ == "__main__":
    
    train_shots = [30207]
    data_names =["magnetics-flux_loop_flux"]
    target_names =["magnetics-flux_loop_flux"]

    import os
    import sys
    
    sys.path.append("./scripts/MAST_tools")
    from MAST_dataset import MastDataset
  
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