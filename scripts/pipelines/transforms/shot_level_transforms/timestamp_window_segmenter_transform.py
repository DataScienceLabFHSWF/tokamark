from typing import List, Dict, Any
import numpy as np


# ======================================================================================================================
class TimestampWindowSegmenterTransform:

    # ------------------------------------------------------------------------------------------------------------------
    def __init__(self, dict_metadata, past_window_sec, future_window_sec, dt_sec=0.0, verbose=False):
        self.dict_metadata = dict_metadata
        self.past_window_sec = past_window_sec
        self.future_window_sec = future_window_sec
        self.dt_sec = dt_sec
        self.verbose = verbose
    
    # ------------------------------------------------------------------------------------------------------------------
    def __call__(self, shot: Dict[str, Any]) -> Dict[str, Any]:

        # Copy non-window metadata
        new_shot = {k: v for k, v in shot.items() if k not in ["x_past", "x_future", "y_past", "y_future"]}
        window_index = shot['window_index']

        # Replace windowed data
        new_shot.update({
            "x_past": self._slice_window(shot["x_past"], self.past_window_sec, "past", window_index),
            "x_future": self._slice_window(shot["x_future"], self.future_window_sec, "future", window_index),
            "y_past": self._slice_window(shot["y_past"], self.past_window_sec, "past", window_index),
            "y_future": self._slice_window(shot["y_future"], self.future_window_sec, "future", window_index),
        })

        return new_shot
    
    # ------------------------------------------------------------------------------------------------------------------
    def _slice_window(self, data: Dict[str, Any], window_sec: float, mode: str, window_index: int) -> Dict[str, Any]:
        """Helper to slice time series data within a given window."""
        new_data = {}
        for key, d_var in data.items():
            # print(key)

            dt = self.dict_metadata[key]['dt'] 
            window_length = int( window_sec / dt )
            shape_values = self.dict_metadata[key]['values_shape']

            if mode == "past":

                # Handle when var not present or not enough datapoint
                times = np.concatenate([np.full(window_length, np.nan), d_var["time"]], axis=-1)
                if d_var["values"].size == 0:
                    values = np.concatenate([np.full(shape_values + (window_length,), np.nan, dtype=float)], axis=-1)
                else:
                    values = np.concatenate([np.full(shape_values + (window_length,), np.nan, dtype=float), d_var["values"]], axis=-1)

                new_data[key] = {
                    "time": times[-window_length:],
                    "values": values[..., -window_length:],
                }
                # print(key, mode, values[..., -window_length:].shape)


            elif mode == "future":

                # Handle when var not present or not enough datapoint
                times = np.concatenate([d_var["time"], np.full(window_length, np.nan)], axis=-1)
                if d_var["values"].size == 0:
                    values = np.concatenate([np.full(shape_values + (window_length,), np.nan, dtype=float)], axis=-1)
                else:
                    values = np.concatenate([d_var["values"], np.full(shape_values + (window_length,), np.nan, dtype=float)], axis=-1)

                new_data[key] = {
                    "time": times[:window_length],
                    "values": values[..., :window_length],
                }
                # print(key, mode, values[..., :window_length].shape)
            else:
                raise ValueError(f"Invalid mode '{mode}' — expected 'past' or 'future'.")

        return new_data

