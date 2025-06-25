"""
window_segmenter_transform.py

This module defines a general-purpose WindowSegmenterTransform class to transform a shot
(a dictionary of time series or profile signals) into a list of time-aligned (x, y) windows
for supervised learning tasks such as regression or forecasting.

Supports:
---------
- Time-based segmentation
- Asymmetric window durations for x and y
- Delays (dt) between x and y
- Stride-based or unitary-step segmentation
- Optional removal of incomplete windows
- Auto-adjustment for zero-length windows
- Channel-wise stacking of multivariate signals
- Consistent alignment across multiple signals
- Optional debug printing

Example usage:
--------------
    segmenter = WindowSegmenterTransform(
        x_keys=["input_signal_1", "input_signal_2"],
        y_keys=["target_signal"],
        x_window_sec=0.02,
        y_window_sec=0.01,
        dt_sec=0.002,
        stride_sec=0.005,
        drop_incomplete_windows=True,
        verbose=True
    )
    windowed_data = segmenter(shot)

Special behaviors:
------------------
- If `stride_unitary = True`:
    The stride between windows is automatically set to the native sampling resolution
    (i.e., the smallest time delta between samples). This produces overlapping, densely sampled windows
    with stride = Δt.

- If `x_window_sec = 0` or `y_window_sec = 0`:
    The corresponding window duration is automatically replaced with one time step
    (i.e., Δt), resulting in a single-sample window for that side. This is useful for
    evaluating instantaneous responses or preparing 1-step-ahead forecasting windows

Input format:
-------------
    shot : dict
        {
            'signal_name': {
                'time': np.ndarray,       # shape (T,)
                'values': np.ndarray      # shape (C, T)
            },
            ...,
            'shot_id': int or str        # optional metadata (non-signal)
        }

Output format:
--------------
    List of windows, each as a dictionary:
        {
            'x': {
                signal_name: {
                    'time': np.ndarray,   # shape (T_x,)
                    'values': np.ndarray  # shape (C, T_x)
                },
                ...
            },
            'y': {
                signal_name: {
                    'time': np.ndarray,   # shape (T_y,)
                    'values': np.ndarray  # shape (C, T_y)
                },
                ...
            },
            'window_index': int,          # sequential window number
            'shot_id': int or str         # carried over from input if present
        }

Notes:
------
- Signal slices are returned as separate entries under 'x' and 'y', organized by signal name.
- All output time arrays are aligned to the original shot's sampling grid.
- This structure supports downstream projection, modeling, and evaluation steps.
"""


import numpy as np


class WindowSegmenterTransform:
    def __init__(
        self,
        x_keys,
        y_keys,
        x_window_sec,
        y_window_sec,
        stride_sec=None,
        dt_sec=0.0,
        min_samples_per_window=1,
        drop_incomplete_windows=True,
        verbose=False,
        stride_unitary=False,
    ):
        self.x_keys = x_keys
        self.y_keys = y_keys
        self.x_window_sec = x_window_sec
        self.y_window_sec = y_window_sec
        self.stride_sec = stride_sec
        self.dt_sec = dt_sec
        self.min_samples = min_samples_per_window
        self.drop_incomplete = drop_incomplete_windows
        self.verbose = verbose
        self.stride_unitary = stride_unitary

    def __call__(self, shot):

        for var, entry in shot.items():
            if not (isinstance(entry, dict) and "values" in entry and "time" in entry):
                continue
            v = entry["values"]
            if v.ndim == 1:
                entry["values"] = v[None, :]
            assert entry["values"].ndim == 2, f"Signal {var} must have shape (C, T)"

        time_ref = shot[self.x_keys[0]]['time']
        start_time = time_ref[0]
        end_time = time_ref[-1]

        delta_t = np.min(np.diff(time_ref))
        if self.x_window_sec == 0:
            if self.verbose:
                print("[INFO] x_window_sec=0 detected → auto-setting to one-sample window using delta_t")
            self.x_window_sec = delta_t
        if self.y_window_sec == 0:
            if self.verbose:
                print("[INFO] y_window_sec=0 detected → auto-setting to one-sample window using delta_t")
            self.y_window_sec = delta_t

        stride = delta_t if self.stride_unitary else self.stride_sec

        required_span = self.x_window_sec + self.dt_sec + self.y_window_sec
        if self.verbose:
            print(f"📦 Required total window span: {required_span:.6f} seconds\n")

        results = []
        t_x_start = start_time
        i = 0

        while True:
            t_x_end = t_x_start + self.x_window_sec
            t_y_start = t_x_end + self.dt_sec
            t_y_end = t_y_start + self.y_window_sec

            if t_y_end > end_time:
                break

            x_data, x_time, x_map = self._collect_time_window(shot, self.x_keys, t_x_start, t_x_end)
            y_data, y_time, y_map = self._collect_time_window(shot, self.y_keys, t_y_start, t_y_end)

            if x_data is not None and y_data is not None:

                x={}
                for var, slice in x_map.items():
                    x[var] = {}
                    x[var]['time'] = x_time
                    # reconstruction of channel
                    x[var]['values'] = x_data[slice]
                y={}
                for var, slice in y_map.items():
                    y[var] = {}
                    y[var]['time'] = y_time
                    # reconstruction of channel
                    y[var]['values'] = y_data[slice]
                result = {
                    'x': x,
                    'y': y,
                    'window_index': i,
                    'shot_id': shot.get("shot_id", None)
                }
                results.append(result)
                if self.verbose:
                    print(f"[Window {i}] x_samples: {len(x_time)}, y_samples: {len(y_time)}")
            elif self.verbose:
                print(f"[Window {i}] skipped due to insufficient samples")

            t_x_start += stride
            i += 1

        return results

    def _collect_time_window(self, shot, keys, t_start, t_end):
        arrays = []
        times_out = None
        channel_map = {}
        channel_index = 0

        for k in keys:
            signal = shot[k]
            times = signal['time']
            values = signal['values']

            mask = (times >= t_start) & (times < t_end)
            idx = np.nonzero(mask)[0]

            if len(idx) < self.min_samples:
                if self.drop_incomplete:
                    return None, None, None
                else:
                    slice_vals = np.full((values.shape[0], self.min_samples), np.nan)
                    slice_time = np.linspace(t_start, t_end, self.min_samples)
            else:
                slice_vals = values[:, idx[0]:idx[-1] + 1]
                slice_time = np.array(times[idx[0]:idx[-1] + 1])

            arrays.append(slice_vals)
            channel_map[k] = slice(channel_index, channel_index + slice_vals.shape[0])
            channel_index += slice_vals.shape[0]

            if times_out is None:
                times_out = slice_time
            else:
                if not np.allclose(times_out, slice_time, atol=1e-8):
                    raise ValueError(f"Inconsistent timestamps for key: {k}")

        data_out = np.concatenate(arrays, axis=0)
        return data_out, times_out, channel_map
