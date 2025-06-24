import os
import sys
cwd = os.path.dirname(os.getcwd())
mother_dir = os.path.dirname(cwd) + os.sep
sys.path.append(os.path.abspath(os.path.join(mother_dir , "MAST_tools")))
sys.path.append(mother_dir)
sys.path.append(cwd)
sys.path.append(os.path.join( cwd,  "scripts" ) )

import pandas as pd
from torch.utils.data import Dataset

from MAST_tools.signal_utils import MASTSignalManager  
from MAST_tools.store_utils import MASTStorageManager



class MAST_Dataset(Dataset):
    """Dataset class for MAST data.

    Parameters
    ----------
    Dataset : torch.utils.data.Dataset
        See __init__ for details.
    """


    def __init__(
        self, 
        local:bool,
        shots_list: list[int], 
        source_signal_list = list,
        variable_transform_map=None,
        shot_transform=None
        ):
        """ Initialize the MASTDataset.
        Parameters
        ----------
        local : bool
            If True, use local MAST database, otherwise use remote S3 bucket
        shots_list : list[int]
            List of shot IDs to load data for.
        source_signal_list : list[str]
            List of data names to load, format: ('source', 'signal')
        variable_transform_map : _type_, optional
            dict map of pipeline of transforms to apply at variable level, by default None
        shot_transform : _type_, optional
            pipeline of transforms to apply at shot level, by default None
        """

        self.local = local
        self.shots_list = shots_list
        self.source_signal_list = source_signal_list
        self.variable_transform_map = variable_transform_map
        self.shot_transform = shot_transform
        self.sig = MASTSignalManager()  
        self.store_manager = MASTStorageManager() # Rodrigo said something about creating twice the same thing by doing this

    def __len__(self):
        return len(self.shots_list)


    def __getitem__(self, idx):

        store = self.store_manager.make_shot_store(
            shot_info = {
                "shot_id":self.shots_list[idx], 
                "local":self.local # Add among attributes
                }
            )

        shot = {}

        # Collect variables (i.e. source-signal) of interest
        for source, signal in self.source_signal_list:

            shot_profile = self.sig.get_profile(
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
            
            # Apply variable-level transforms
            if self.variable_transform_map and (shot_vals is not None and shot_time is not None):
                shot[f'{source}-{signal}'] = self.variable_transform_map[f'{source}-{signal}']({"time": shot_time, "values": shot_vals})
            else:
                shot[f'{source}-{signal}'] = {"time": shot_time, "values": shot_vals}
            
        # Apply shot-level transforms to obtain a list of training objects
        if self.shot_transform :
            if all( subval is not None 
                for subdict in shot.values() 
                for subval in subdict.values()) :
                list_chunks = self.shot_transform( shot )
                return list_chunks
            else:
                print('Nan still present in shot')
                return []
        else:
            return(shot)