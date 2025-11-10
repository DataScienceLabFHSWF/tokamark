from typing import List, Dict, Any
import numpy as np


# ======================================================================================================================
class RollingSegmenterTransform:

    # ------------------------------------------------------------------------------------------------------------------
    def __init__(
        self,
        x_past_keys,
        x_future_keys = None,
        y_past_keys = None,
        y_future_keys = None,
        verbose=False,
    ):

        self.x_past_keys = [
        f"{source}-{signal}" for source, signal in (x_past_keys or [])
        ]
        self.x_future_keys = [
            f"{source}-{signal}" for source, signal in (x_future_keys or [])
        ]
        self.y_past_keys = [
            f"{source}-{signal}" for source, signal in (y_past_keys or [])
        ]
        self.y_future_keys = [
            f"{source}-{signal}" for source, signal in (y_future_keys or [])
        ]

        self.verbose = verbose

    # ------------------------------------------------------------------------------------------------------------------
    def __call__(self, shot: Dict[str, Any]) -> List[Dict[str, Any]]:
        
        # Raise Error if some Nans still present in the time vector (fallback but it should have been checked before)
        # for var, entry in shot.items():
        #     if np.isnan(entry["time"]).any():
        #         raise ValueError(f"[ERROR] Signal '{var}' contains NaN values in its 'time' array.")

        t_start = []
        t_end = []
        delta_ts = []
        for var in self.x_past_keys + self.x_future_keys + self.y_past_keys + self.y_future_keys :
            t = shot[var]["time"]
            if t.size != 0:
                t_start.append(t[0])
                t_end.append(t[-1])
                dts = np.diff(t)
                if len(dts) > 0:
                    delta_ts.append(np.min(dts))
                    if self.verbose:
                        print(f"\nt_start for {var}: {t[0]:.6f}")
                        print(f"t_end for {var}: {t[-1]:.6f} s")
                        print(f"Δt for {var}: {np.min(dts):.6f} s")
        if not delta_ts:
            raise ValueError("No valid Δt found in any signals.")  

        start_time = np.min(t_start)
        end_time = np.max(t_end)


        min_dt = np.min(delta_ts)
        max_dt = np.max(delta_ts)
        stride = max_dt

        results = []
        t_cut = start_time
        i = 0

        while t_cut < end_time :

            x_past = self._collect_past_per_signal_windows(shot, self.x_past_keys, t_cut)
            x_future = self._collect_future_per_signal_windows(shot, self.x_future_keys, t_cut)

            y_past = self._collect_past_per_signal_windows(shot, self.y_past_keys, t_cut)
            y_future = self._collect_future_per_signal_windows(shot, self.y_future_keys, t_cut)

            result = {
                    "window_index": i,
                    "t_cut": t_cut,
                    "x_past": x_past,
                    "x_future": x_future,
                    "y_past": y_past,
                    "y_future": y_future,
            }
            results.append(result)

            t_cut += stride
            i += 1

        return results

    # ------------------------------------------------------------------------------------------------------------------
    def _collect_past_per_signal_windows(self, shot, keys, t_cut):
        signal_slices = {}
        for key in keys:
            signal = shot[key]
            times = signal["time"]
            values = signal["values"]
            mask = (times < t_cut)
            
            sliced_vals = values[..., mask]
            sliced_time = times[mask]

            signal_slices[key] = {
                "time": sliced_time,
                "values": sliced_vals
            }

        return signal_slices
    
    # ------------------------------------------------------------------------------------------------------------------
    def _collect_future_per_signal_windows(self, shot, keys, t_cut):
        signal_slices = {}
        for key in keys:
            signal = shot[key]
            times = signal["time"]
            values = signal["values"]
            mask = (times >= t_cut)
            
            sliced_vals = values[..., mask]
            sliced_time = times[mask]

            signal_slices[key] = {
                "time": sliced_time,
                "values": sliced_vals
            }

        return signal_slices

    # ------------------------------------------------------------------------------------------------------------------
