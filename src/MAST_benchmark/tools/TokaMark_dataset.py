import random
import numpy as np
from torch.utils.data import IterableDataset, get_worker_info
from typing import Optional, Mapping, Any


# ----------------------------------------------------------------------------------------------------------------------
# HELPERS
# ----------------------------------------------------------------------------------------------------------------------

# --------------------------------------------------------------
# Filtering
# --------------------------------------------------------------
def _all_vars_have_nans(dict_obj):
    return all([np.isnan(np.asarray(dict_obj[var]['values'])).any() for var in dict_obj.keys()])

def _any_vars_have_nans(dict_obj):
    return any([np.isnan(np.asarray(dict_obj[var]['values'])).any() for var in dict_obj.keys()])

# --------------------------------------------------------------
# Optional streaming shuffle buffer
# --------------------------------------------------------------
def _shuffle_buffer(iterator, buffer_size=512):
    buffer = []
    for item in iterator:
        buffer.append(item)
        if len(buffer) >= buffer_size:
            idx = random.randrange(len(buffer))
            yield buffer.pop(idx)
    while buffer:
        idx = random.randrange(len(buffer))
        yield buffer.pop(idx)


# ----------------------------------------------------------------------------------------------------------------------
# MAIN DATASET
# ----------------------------------------------------------------------------------------------------------------------

class TokaMarkDataset(IterableDataset):

    def __init__(
        self,
        base_dataset,
        task_metadata: Mapping[str, Any],
        config_metadata: Mapping[str, Any],
        custom_transform: Optional[Any] = None,
        test_mode: bool = False,
        shuffle_windows: bool = False,
        shuffle_buffer_size: int = 512,
        verbose: bool = False,
    ):
        super().__init__()

        self.base = base_dataset
        self.shots_list = self.base.shots_list
        self.task_metadata = task_metadata

        self.data_metadata = {
            key: {'dt': meta['dt'], 'shape_values': tuple(meta['values_shape'])}
            for d in [task_metadata['input'], task_metadata['actuator'], task_metadata['output']]
            for key, meta in d.items()
        }

        self.task_type = config_metadata["task_type"]
        seg = config_metadata["task_window_segmenter"]

        self.input_keys = [f"{s}-{k}" for s, k in (seg["input_keys"] or [])]
        self.actuator_keys = [f"{s}-{k}" for s, k in (seg["actuator_keys"] or [])]
        self.output_keys = [f"{s}-{k}" for s, k in (seg["output_keys"] or [])]

        self.input_length = seg["input_length"]
        self.output_length = seg["output_length"]
        self.delta = seg["delta"]

        self.stride = float(self.task_metadata["sec_stride"])

        self.custom_transform = custom_transform
        self.test_mode = test_mode
        self.shuffle_windows = shuffle_windows
        self.shuffle_buffer_size = shuffle_buffer_size
        self.verbose = verbose

    # ------------------------------------------------------------------------------------------------------------------
    # Worker-safe iterator
    # ------------------------------------------------------------------------------------------------------------------
    def __iter__(self):

        worker_info = get_worker_info()

        if worker_info is None:
            start = 0
            end = len(self.base)
        else:
            per_worker = int(np.ceil(len(self.base) / worker_info.num_workers))
            start = worker_info.id * per_worker
            end = min(start + per_worker, len(self.base))

        iterator = self._iterate_shots(start, end)

        if self.shuffle_windows:
            iterator = _shuffle_buffer(iterator, self.shuffle_buffer_size)

        yield from iterator

    # ------------------------------------------------------------------------------------------------------------------
    # Iterate function
    # ------------------------------------------------------------------------------------------------------------------
    def _iterate_shots(self, start, end):

        for idx_shot in range(start, end):
            yield from self._process_shot(idx_shot)

    # ------------------------------------------------------------------------------------------------------------------
    # Shot processing
    # ------------------------------------------------------------------------------------------------------------------
    def _process_shot(self, idx_shot):

        sample = self.base[idx_shot]

        # --------------------------------------------------------------
        # determine valid global time range (from outputs)
        # --------------------------------------------------------------
        t_start, t_end, delta_ts = [], [], []

        for var in self.output_keys:
            t = sample[var]["time"]
            v = sample[var]["values"]

            if t.size == 0 or v.size == 0:
                continue

            # True where timestep has at least one real value
            valid_mask = ~np.all(np.isnan(v), axis=tuple(range(v.ndim - 1)))
            if not np.any(valid_mask):
                continue
            t_valid = t[valid_mask]
            t_start.append(t_valid[0])
            t_end.append(t_valid[-1])
            dts = np.diff(t_valid)
            if dts.size:
                delta_ts.append(np.min(dts))
        if not delta_ts:
            return

        global_start_time = np.min(t_start)
        global_end_time = np.max(t_end)

        # Pad sample here 
        sample = self._pad_sample_to_interval(
            sample,
            global_start_time,
            global_end_time,
        )

        t_cuts = np.arange(global_start_time + self.input_length, 
                           global_end_time - self.delta - self.output_length, 
                           self.stride)

        # --------------------------------------------------------------
        # build windows
        # --------------------------------------------------------------
        for idx_t, t_cut in enumerate(t_cuts):
            
            input_slice = self._build_window(sample, 
                                             global_start_time,
                                             t_cut, 
                                             "input")
            actuator_slice = self._build_window(sample, 
                                                global_start_time, 
                                                t_cut, 
                                                "actuator")
            output_slice = self._build_window(sample, 
                                              global_start_time,
                                              t_cut, 
                                              "output")

            obj = {
                "input": input_slice,
                "actuator": actuator_slice,
                "output": output_slice,
                "t_cut": t_cut,
                "shot_id": self.get_shot_id(idx_shot),
                "window_index": idx_t,
            }

            # ----------------------------------------------------------
            # filtering
            # ----------------------------------------------------------
            if self.test_mode:
                window_valid = (
                    not (
                        _all_vars_have_nans(obj["input"])
                        and _all_vars_have_nans(obj["actuator"])
                    )
                    and not _any_vars_have_nans(obj["output"])
                )
                if not window_valid:
                    print(f"Window {obj['window_index']} of shot {obj['shot_id']} is not valid")
                    continue

            obj2 = self.custom_transform(obj) if self.custom_transform else obj
            if obj2 is None:
                continue

            yield {
                "shot_id": obj["shot_id"],
                "window_index": idx_t,
                **obj2,
            }

    # ------------------------------------------------------------------------------------------------------------------
    # Window builder
    # ------------------------------------------------------------------------------------------------------------------
    def _build_window(self, 
                      sample, 
                      global_start_time,
                      t_cut,
                      type_window # "input", "actuator", "output"
                      ):

        out = {}

        keys = {
            "input": self.input_keys,
            "actuator": self.actuator_keys,
            "output": self.output_keys,
        }

        for key in keys[type_window]:

            md = self.task_metadata[type_window][key]

            dt = md["dt"]
            shape_values = tuple(md["values_shape"])

            ts_in = int(round(self.input_length / dt))
            ts_out = int(round(self.output_length / dt))
            ts_delta = int(round(self.delta / dt))

            ts_len = {
                "input": ts_in,
                "actuator": ts_in + ts_delta + ts_out, 
                "output": ts_out
            }

            times = sample[key]["time"]
            values = sample[key]["values"]

            if times.size==0 or values.size==0:
                selected_times = np.full(ts_len[type_window], np.nan)
                selected_values = np.full(shape_values + (ts_len[type_window],), np.nan)

            else:

                # --------------------------------------------------
                # Index Logic
                # --------------------------------------------------
                if type_window == "input":
                    end_idx = int(round((t_cut - times[0]) / dt))
                    if self.task_type == "markovian":
                        idx = np.arange(end_idx - ts_in, end_idx)
                    else:
                        global_start_idx = int(round((global_start_time - times[0]) / dt))
                        idx = np.arange(global_start_idx, end_idx)

                elif type_window == "output":
                    cut_time = t_cut + self.delta
                    start_idx = int(round((cut_time - times[0]) / dt))
                    idx = np.arange(start_idx, start_idx + ts_out)

                elif type_window == "actuator":
                    cut_idx = int(round((t_cut - times[0]) / dt))
                    if self.task_type == "markovian":
                        idx_in = np.arange(cut_idx - ts_in, cut_idx)
                    else:
                        global_start_idx = int(round((global_start_time - times[0]) / dt))
                        idx_in = np.arange(global_start_idx, cut_idx)
                    idx_out = np.arange(
                        cut_idx,
                        cut_idx + ts_delta + ts_out
                    )

                    idx = np.concatenate([idx_in, idx_out])
                # --------------------------------------------------
                
                idx = np.clip(idx, 0, len(times) - 1)
                selected_times = times[idx]
                selected_values = values[..., idx]

            if selected_values.ndim == 2 and selected_values.shape[0] == 1:
                selected_values = selected_values[0]

            out[key] = {"time": selected_times, "values": selected_values}

        return out

    # ------------------------------------------------------------------------------------------------------------------
    # Padding
    # ------------------------------------------------------------------------------------------------------------------
    def _pad_timeseries_to_interval(self, times, values, dt, t_start, t_end, shape_values):
        
        if not np.issubdtype(values.dtype, np.floating):
            values = values.astype(float)
        
        # ---------- LEFT PAD ----------
        if times[0] > t_start:
            n_pad = int(np.ceil((times[0] - t_start) / dt)) + 1

            left_times = times[0] - dt * np.arange(n_pad, 0, -1)
            times = np.concatenate([left_times, times])

            pad_shape = shape_values + (n_pad,)
            left_pad = np.full(pad_shape, np.nan)

            values = np.concatenate([left_pad, values], axis=-1)

        # ---------- RIGHT PAD ----------
        if times[-1] < t_end:
            n_pad = int(np.ceil((t_end - times[-1]) / dt)) + 1

            right_times = times[-1] + dt * np.arange(1, n_pad + 1)
            times = np.concatenate([times, right_times])

            pad_shape = shape_values + (n_pad,)
            right_pad = np.full(pad_shape, np.nan)

            values = np.concatenate([values, right_pad], axis=-1)

        return times, values

    def _pad_sample_to_interval(self, sample, t_start, t_end):

        for key in sample.keys():
            times = sample[key]["time"]
            values = sample[key]["values"]

            if times.size == 0 or values.size == 0:
                continue

            dt = self.data_metadata[key]['dt']
            shape_values = self.data_metadata[key]['shape_values']

            times, values = self._pad_timeseries_to_interval(times, values, dt, t_start, t_end, shape_values)

            sample[key]["time"] = times
            sample[key]["values"] = values

        return sample

    # ------------------------------------------------------------------------------------------------------------------
    # Get shot ID
    # ------------------------------------------------------------------------------------------------------------------
    def get_shot_id(self, idx):
        return self.base.shots_list[idx]
    