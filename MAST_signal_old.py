import MAST_store as MAST
import xarray as xr
import pandas as pd
import zarr
import s3fs
import fsspec
from concurrent.futures import ThreadPoolExecutor, as_completed

"""
Class SIGNAL to process signals from fair MAST bucket
"""

class SIGNAL:
    def __init__(self, group, signal_name, shot_ids):
        self.group = group  
        self.signal_name = signal_name
        self.shot_ids = shot_ids
       

    def get_profile_data(self,store):
        """
        Get time and data from a signal profile.
        """
        data = None
        time = None

        try:
            profile = xr.open_zarr(store, group=self.group)
        except Exception as e:
            print(f"Error opening Zarr store: {e}")
            return None, None

        try:
            data = profile[self.signal_name]
        except KeyError:
            print(f"Signal '{self.signal_name}' not found in group '{self.group}'.")
        except Exception as e:
            print(f"Error accessing signal '{self.signal_name}': {e}")

        try:
            time = profile["time"]
        except KeyError:
            print("Time coordinate not found.")
        except Exception as e:
            print(f"Error accessing time: {e}")

        return data, time

    def get_values(self, store):
        try:
            profile = xr.open_zarr(store, group=self.group)
            return profile[self.signal_name].values
        except Exception as e:
            print(f"Error opening Zarr store: {e}")
            return None

    def all_profiles_to_dataFrame(self):
        """
        Dump all signals from store into a pandas dataFrame.
        """
        if len(shot_ids < 2):
            print("You must pass at least two shot_ids")
            return

        rows = []

        tot_ids = len(self.shot_ids)
        for nr, shot_id in enumerate(self.shot_ids):
            print(f"{nr} out of {tot_ids}")
            store = MAST.MAST_make_store(shot_id)
            data, time = self.get_signal(store, self.group, self.signal)

            if data is not None and time is not None:
                rows.append({
                    "shot_id": shot_id,
                    "type": "data",
                    "values": data.values
                })
                rows.append({
                    "shot_id": shot_id,
                    "type": "time",
                    "values": time.values
                })

        df = pd.DataFrame(rows)
        return df
    

    def all_profiles_to_dataFrame_in_parallel(self, max_workers=4):
        """
        Dump all signals from store into a pandas dataFrame. Use parallelization.
        """
        if len(shot_ids < 2):
            print("You must pass at least two shot_ids")
            return

        def process_shot(shot_id):
            try:
                store = MAST.MAST_make_store(shot_id)
                data, time = MAST.get_signal(store, self.group, self.signal)
                if data is not None and time is not None:
                    return [
                        {"shot_id": shot_id, "type": "data", "values": data.values},
                        {"shot_id": shot_id, "type": "time", "values": time.values},
                    ]
            except Exception as e:
                print(f"Error processing shot {shot_id}: {e}")
            return []

        rows = []
        tot_ids = len(self.shot_ids)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(process_shot, sid): sid for sid in self.shot_ids}

            for i, future in enumerate(as_completed(futures)):
                shot_id = futures[future]
                try:
                    result = future.result()
                    rows.extend(result)
                except Exception as e:
                    print(f"Shot {shot_id} failed: {e}")
                print(f"{i+1}/{tot_ids} done")

        df = pd.DataFrame(rows)
        return df
    


def test():
    group = "magnetics"
    signal_name = "flux_loop_flux"

    shot = 30421
    sig = SIGNAL(group, signal_name, shot)

    store = MAST.MAST_make_store( shot)

    sig.get_data(store)

    sig.get_values(store)
    breakpoint()

    print("end")

if __name__ == "__main__":
    test()

