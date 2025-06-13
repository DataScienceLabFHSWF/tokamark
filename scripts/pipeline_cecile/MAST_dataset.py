import os
import pandas as pd
import sys

from torch.utils.data import Dataset

cwd = os.path.dirname(os.getcwd())
mother_dir = os.path.dirname(cwd) + os.sep
sys.path.append(os.path.abspath(os.path.join(mother_dir , "MAST_tools")))
sys.path.append(mother_dir)
sys.path.append(cwd)
sys.path.append(os.path.join( os.path.dirname(cwd) ) )

from MAST_tools.signal_utils import MASTSignalManager  
from MAST_tools.store_utils import MASTStorageManager



class MAST_Dataset(Dataset):
    """
    Custom Dataset for loading MAST data
    """

    def __init__(
        self, 
        local:bool,
        shots_list: list[int], 
        source_signal_list = list,
        transform_map=None,
        model_specific_transform=None
        ):

        self.local = local
        self.shots_list = shots_list
        self.source_signal_list = source_signal_list
        self.transform_map = transform_map
        self.model_specific_transform = model_specific_transform


    def __len__(self):
        return len(self.shots_list)


    def __getitem__(self, idx):

        try:

            sig = MASTSignalManager()  
            
            store_manager =  MASTStorageManager()

            store = store_manager.make_shot_store(
                shot_info = {
                    "shot_id":self.shots_list[idx], 
                    "local":self.local # Add among attributes
                    }
                )
            
            # loop on groups of x

            shot = {}

            # print(self.source_signal_x_list)
            for source, signal in self.source_signal_list:

                shot_profile = sig.get_profile(
                        data_origin=store,
                        source_name=source,
                        signal_name=signal
                        )
                
                if shot_profile is not None:
                    try:
                        shot_time = shot_profile.time.values
                    except  AttributeError:
                        shot_time = None
                    try:
                        shot_vals = shot_profile.values
                    except  AttributeError:
                        shot_vals = None
                else:
                    shot_vals = None
                    shot_time = None
                
                # fill the tensordict
                # if self.transform_map_x and (x_vals.size >0 and x_time.size >0):
                if self.transform_map and (shot_vals is not None and shot_time is not None):
                    #print(source, signal)
                    #print('Apply transform because full shot')
                    shot[f'{source}-{signal}'] = self.transform_map[f'{source}-{signal}']({"time": shot_time, "values": shot_vals})
                else:
                    #print('Empty shot')
                    shot[f'{source}-{signal}'] = {"time": shot_time, "values": shot_vals}
                
                # Apply Model-specific transforms to obtain a list of my training objects

            if self.model_specific_transform :
                list_chunks = []
                if all( subval is not None 
                    for subdict in shot.values() 
                    for subval in subdict.values()) :
                    # print('No None in shot')
                    list_x, list_y = self.model_specific_transform( shot )
                    for xc, yc in zip(list_x, list_y):
                        list_chunks.append( (xc, yc) )
                return list_chunks
            
            else:
                return(shot)

        except Exception as e:
            print(f"🔥 Error in __getitem__ at idx={idx}: {e}")
            raise