import MAST_data_visualizer as fairMAST
import os
import pandas as pd
import zarr
import s3fs
import fsspec
from concurrent.futures import ThreadPoolExecutor, as_completed

"""
Class SIGNAL to extract signal profiles from fair MAST bucket
"""

class SIGNAL:
    def __init__(self, group, signal, shot_ids):
        self.group = group  
        self.signal = signal
        self.shot_ids = shot_ids
       

    def all_profiles_to_dataFrame(self):
        rows = []

        tot_ids = len(self.shot_ids)
        for nr, shot_id in enumerate(self.shot_ids):
            print(f"{nr} out of {tot_ids}")
            store = fairMAST.MAST_make_store(shot_id)
            data, time = fairMAST.get_signal(store, self.group, self.signal)

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
    

    def all_profiles_to_dataFrame_2(self, max_workers=10):
        def process_shot(shot_id):
            try:
                store = fairMAST.MAST_make_store(shot_id)
                data, time = fairMAST.get_signal(store, self.group, self.signal)
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
    group = "summary"
    signal_name = "palsma_current"

    shots = fairMAST.list_all_shots()
    shot_ids = [int(os.path.basename(url).replace(".zarr","")) for url in shots["url"]]
    current = SIGNAL(group, signal_name, shot_ids)
    df = current.all_profiles_to_dataFrame_2()
    breakpoint()
    df.to_excel("shot_data.xlsx", sheet_name=signal_name, index=False)


if __name__ == "__main__":
    test()

