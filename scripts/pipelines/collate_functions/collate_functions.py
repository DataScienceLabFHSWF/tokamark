
"""This file contains the collate functions for batching samples in a DataLoader. 
    
    Expected output of the collate function is a dictionary with two keys: 
        return {'x': batched_x, 'y': batched_y}
    
    see transforms.SegmenterTransform for more details on the input data format.
"""      

def first_item(batch):
    return batch[0]
    
class TimeWindowSegmentationCollateFn:
    def __call__(self, list_x, list_y):
        all_x_segments = []
        all_y_segments = []
        
        len_x, len_y = len(list_x), len(list_y)
        
        if len_x is None or len_y is None:
            print("Warning: problem with data lengths after segmentation: list_x or list_y is None")
            return None

        if len_y == 0 or len_x == 0:
            print("Warning: problem with data lengths after segmentation")
            print(f"Lengths of lists: len(list_y) = {len(list_y)}, len(list_x) = {len(list_x)}")
            return None
        
        # If the lengths of the lists are not equal, we truncate them removing 
        if len_x != len_y:
            if len_x > len_y:
                list_x = list_x[len_x - len_y:]
            elif len_y > len_x:
                list_y = list_y[len_y - len_x:]
            
        for x_segment, y_segment in zip(list_x, list_y):
            # Extract x values
            x_values = [
                signal_dict["values"]
                for signal_dict in x_segment["sources_signals"]
            ]
            # shape: [num_signals, nr_features, time_window_length]
            
            # Extract y values
            y_values = [
                signal_dict["values"]
                for signal_dict in y_segment["sources_signals"]
            ]
            # shape: [num_signals, nr_features, time_window_length]
            
            all_x_segments.append(x_values) # shape: [list_x_length, num_signals, nr_features, time_window_length]
            all_y_segments.append(y_values) # shape: [list_y_length, num_signals, nr_features, time_window_length]
        
        if not all_x_segments or not all_y_segments:
            return None  # or return empty batch dicts
        
        return {'x': all_x_segments, 'y': all_y_segments}