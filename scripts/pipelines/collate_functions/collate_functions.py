from transforms.signal_level_transforms.segmenter_transform import SegmenterTransform

"""This file contains the collate functions for batching samples in a DataLoader. 
    
    Expected output of the collate function is a dictionary with two keys: 
        return {'x': batched_x, 'y': batched_y}
    
    see transforms.SegmenterTransform for more details on the input data format.
"""      

def first_item(batch):
    return batch[0]

class MiniBatchCollateFn:
    
    def __init__(self,
                data_names,
                target_names,
                time_window_sec,
                time_step,
                offset):
        
        self.data_names = data_names
        self.target_names = target_names
        self.time_window_sec = time_window_sec
        self.time_step = time_step
        self.offset = offset
        
        self.segmenter_transform = SegmenterTransform(
            data_names=self.data_names,
            target_names=self.target_names,
            time_window_sec=self.time_window_sec, 
            time_step=self.time_step, 
            offset=self.offset
        )

    def __call__(self, batch):
        return self.segmenter_transform(batch)
    
    
class TimeWindowSegmentationCollateFn:
    def __call__(self, list_x, list_y):
        all_x_segments = []
        all_y_segments = []
        breakpoint()
        if list_x is None or list_y is None:
            print("Warning: problem with data lengths after segmentation: list_x or list_y is None")
            return None
        
        if len(list_x) != len(list_y):
            print("Warning: problem with data lengths after segmentation: lengths of list_x and list_y differ")
            print(f"Lengths of lists: len(list_y) = {len(list_y)}, len(list_x) = {len(list_x)}")
            return None

        if len(list_y) == 0 or len(list_x) == 0:
            print("Warning: problem with data lengths after segmentation")
            print(f"Lengths of lists: len(list_y) = {len(list_y)}, len(list_x) = {len(list_x)}")
            return None
                    
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