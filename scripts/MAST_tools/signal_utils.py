# @Author: Rodrigo Ordonez-Hurtado (rodrigo.ordonez.hurtado@ibm.com)

import store_utils
import xarray as xr


# ======================================================================================================================
class MASTSignalManager:
    """
    Class to process signals from fair MAST bucket
    """

    # ------------------------------------------------------------------------------------------------------------------
    def __init__(self, source, signal_name, shot_id):
        self.source = source
        self.signal_name = signal_name
        self.shot_id = shot_id

    # ------------------------------------------------------------------------------------------------------------------
    def get_values(self, store):
        try:
            profile = xr.open_zarr(store, group=self.source)
            return profile[self.signal_name].values
        except Exception as e:
            print(f"Error opening Zarr store: {e}")
            return None

    def get_channel_names(self, store):
        try:
            profile = xr.open_zarr(store, group=self.source)
            data_array = profile[self.signal_name]
            non_time_coords = [coord for coord in data_array.coords if coord != "time"]
            if non_time_coords:
                channel_coord = non_time_coords[0]
                return data_array.coords[channel_coord].values
            else:
                return None
        except Exception as e:
            print(f"Error opening Zarr store: {e}")
            return None

# ----------------------------------------------------------------------------------------------------------------------
def test():
    store_manager = store_utils.MASTStorageManager()

    source = "magnetics"
    signal_name = "flux_loop_flux"
    shot_id = 30421
    signal = MASTSignalManager(source=source, signal_name=signal_name, shot_id=shot_id)
    store_from_shot_id = store_manager.make_shot_store(shot_id=shot_id)
    print(f"Signal values: {signal.get_values(store=store_from_shot_id)}\n")


# ======================================================================================================================
if __name__ == "__main__":
    test()
