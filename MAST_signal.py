import MAST_store as MAST
import xarray as xr
import pandas as pd
import zarr
import s3fs
import fsspec
from concurrent.futures import ThreadPoolExecutor, as_completed

"""
Class to process signals from fair MAST bucket
"""

class SIGNAL:
    def __init__(self, group, signal_name, shot_id):
        self.group = group  
        self.signal_name = signal_name
        self.shot_id = shot_id
       

    def get_values(self, store):
        try:
            profile = xr.open_zarr(store, group=self.group)
            return profile[self.signal_name].values
        except Exception as e:
            print(f"Error opening Zarr store: {e}")
            return None



def test():
    group = "magnetics"
    signal_name = "flux_loop_flux"

    shot = 30421
    sig = SIGNAL(group, signal_name, shot)

    store = MAST.make_store( shot)

    sig.get_values(store)


if __name__ == "__main__":
    test()

