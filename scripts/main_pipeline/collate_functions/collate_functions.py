from transforms.signal_level_transforms.segmenter_transform import SegmenterTransform

"""This file contains the collate functions for batching samples in a DataLoader. 
    
    Expected output of the collate function is a dictionary with two keys: 
        return {'x': batched_x, 'y': batched_y}
    
    see transforms.SegmenterTransform for more details on the input data format.
"""      
class MiniBatchCollateFn:
    
    def __init__(self,
                 time_window_sec,
                 time_step,
                 offset):
        
        self.time_window_sec = time_window_sec
        self.time_step = time_step
        self.offset = offset
        self.segmenter_transform = SegmenterTransform(
            time_window_sec=self.time_window_sec, 
            time_step=self.time_step, 
            offset=self.offset
        )

    def __call__(self, batch):
        return self.segmenter_transform(batch)