import numpy as np
import torch
from torch.utils.data import IterableDataset, get_worker_info
from typing import Optional, Mapping, Any


# --------------------------------------------------------------------------------------
# helpers 
# --------------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------------------------------------------
def all_vars_have_nans(
        dict_obj: Mapping
) -> bool:
    """
    Check if bool(x) is True for all values x in `dict_obj`.

    Parameters
    ----------
    dict_obj : Mapping
        Input mapping.

    Returns
    -------
    bool
        If the iterable is empty, return True.

    """
    return all([np.isnan(np.asarray(dict_obj[var]['values'])).any() for var in dict_obj.keys()])


# ----------------------------------------------------------------------------------------------------------------------
def any_vars_have_nans(
        dict_obj: Mapping
):
    """
    Check if bool(x) is True for any values x in `dict_obj`.

    Parameters
    ----------
    dict_obj
        Input mapping.

    Returns
    -------
    bool
        If the iterable is empty, return True.

    """
    return any([np.isnan(np.asarray(dict_obj[var]['values'])).any() for var in dict_obj.keys()])


# --------------------------------------------------------------------------------------
# optional streaming shuffle buffer
# --------------------------------------------------------------------------------------

def shuffle_buffer(iterator, buffer_size=512):
    """
    Streaming shuffle for IterableDataset.
    """
    import random
    buffer = []

    for item in iterator:
        buffer.append(item)
        if len(buffer) >= buffer_size:
            idx = random.randrange(len(buffer))
            yield buffer.pop(idx)

    while buffer:
        idx = random.randrange(len(buffer))
        yield buffer.pop(idx)



import numpy as np

def pad_timeseries_to_interval(
    times,
    values,
    dt,
    t_start,
    t_end,
    shape_values,
):
    """
    Pad a timeseries so that [t_start, t_end] lies inside `times`.

    Parameters
    ----------
    times : np.ndarray
        1D time array.
    values : np.ndarray
        Data array with time on the last axis.
    dt : float
        Sampling interval.
    t_start : float
        Left boundary that must exist in times.
    t_end : float
        Right boundary that must exist in times.
    shape_values : tuple
        Shape of value dimensions excluding time axis.

    Returns
    -------
    times, values : padded arrays
    """

    # ensure float dtype once (needed for NaNs)
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


# --------------------------------------------------------------------------------------
# MAIN DATASET
# --------------------------------------------------------------------------------------

class TaskModelTransformWrapperIterable(IterableDataset):

    def __init__(
        self,
        base_dataset,
        dict_task_metadata: Mapping[str, Any],
        config_task: Mapping[str, Any],
        model_transform: Optional[Any] = None,
        test_mode: bool = False,
        shuffle_windows: bool = False,
        shuffle_buffer_size: int = 512,
        verbose: bool = False,
    ):
        super().__init__()

        self.base = base_dataset
        self.shots_list = self.base.shots_list
        self.dict_task_metadata = dict_task_metadata

        self.task_type = config_task["task_type"]

        seg = config_task["task_window_segmenter"]

        self.input_keys = [f"{s}-{k}" for s, k in (seg["input_keys"] or [])]
        self.actuator_keys = [f"{s}-{k}" for s, k in (seg["actuator_keys"] or [])]
        self.output_keys = [f"{s}-{k}" for s, k in (seg["output_keys"] or [])]

        self.input_length = seg["input_length"]
        self.output_length = seg["output_length"]
        self.delta = seg["delta"]

        self.stride = float(self.dict_task_metadata["sec_stride"])

        self.model_transform = model_transform
        self.test_mode = test_mode
        self.shuffle_windows = shuffle_windows
        self.shuffle_buffer_size = shuffle_buffer_size
        self.verbose = verbose

    # ------------------------------------------------------------------
    # worker-safe iterator
    # ------------------------------------------------------------------
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
            iterator = shuffle_buffer(iterator, self.shuffle_buffer_size)

        yield from iterator

    # ------------------------------------------------------------------
    def _iterate_shots(self, start, end):

        for idx_shot in range(start, end):
            yield from self._process_shot(idx_shot)

    # ------------------------------------------------------------------
    def _process_shot(self, idx_shot):

        # print('\n')
        # print(self.get_shot_id(idx_shot))

        sample = self.base[idx_shot]

        # --------------------------------------------------------------
        # determine valid global time range (from outputs)
        # --------------------------------------------------------------
        t_start, t_end, delta_ts = [], [], []

        for var in self.output_keys:
            t = sample[var]["time"]
            if t.size:
                t_start.append(t[0])
                t_end.append(t[-1])
                dts = np.diff(t)
                if len(dts):
                    delta_ts.append(np.min(dts))

        if not delta_ts:
            return

        start_time = np.min(t_start)
        end_time = np.max(t_end)

        t_cuts = np.arange(start_time, end_time, self.stride)

        # --------------------------------------------------------------
        # build windows
        # --------------------------------------------------------------
        for idx_t, t_cut in enumerate(t_cuts):
            
            # print(idx_t)

            if self.task_type == "non_markovian":
                # print("Non Markovian")
                input_slice = self._build_non_markovian_input(sample, t_cut)
                actuator_slice = self._build_non_markovian_actuator(sample, t_cut)
            else:
                # print("Markovian")
                input_slice = self._build_input(sample, t_cut)
                actuator_slice = self._build_actuator(sample, t_cut)

            output_slice = self._build_output(sample, t_cut)

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
                # print("test_mode on")
                window_valid = (
                    not (
                        all_vars_have_nans(obj["input"])
                        and all_vars_have_nans(obj["actuator"])
                    )
                    and not any_vars_have_nans(obj["output"])
                )
            else:
                window_valid = True

            if not window_valid:
                # print("window not valid")
                continue
            # else:
            #     print("window valid")

            obj2 = self.model_transform(obj) if self.model_transform else obj
            if obj2 is None:
                continue

            yield {
                "shot_id": obj["shot_id"],
                "window_index": idx_t,
                **obj2,
            }

    # ------------------------------------------------------------------
    # INPUT
    # ------------------------------------------------------------------
    def _build_input(self, sample, t_cut):

        out = {}

        for key in self.input_keys:
            md = self.dict_task_metadata["input"][key]
            ts_len = md["ts_length"]
            dt = md["dt"]
            shape_values = tuple(md["values_shape"])

            times = sample[key]["time"]
            values = sample[key]["values"]

            if times.size==0 or values.size==0:
                selected_times = np.full(ts_len, np.nan)
                selected_values = np.full(shape_values + (ts_len,), np.nan)

            else:
                times, values = pad_timeseries_to_interval(
                    times,
                    values,
                    dt,
                    t_cut - self.input_length, # t_start
                    t_cut, # t_end
                    shape_values,
                )

                # idx = np.where(times < t_cut)[0][-ts_len:]
                cut_idx = np.searchsorted(times, t_cut, side="left")
                idx = np.arange(cut_idx - ts_len, cut_idx)
                # chosen = idx[-ts_len:]
                # print('in', len(idx), idx)
                
                selected_times = times[idx]
                selected_values = values[..., idx]

            # print(selected_values)

            if selected_values.ndim == 2 and selected_values.shape[0] == 1:
                selected_values = selected_values[0]

            out[key] = {"time": selected_times, "values": selected_values}

        return out

    # ------------------------------------------------------------------
    # OUTPUT
    # ------------------------------------------------------------------
    def _build_output(self, sample, t_cut):

        out = {}

        for key in self.output_keys:
            md = self.dict_task_metadata["output"][key]
            ts_len = md["ts_length"]
            dt = md["dt"]
            shape_values = tuple(md["values_shape"])

            times = sample[key]["time"]
            values = sample[key]["values"]

            if times.size==0 or values.size==0:
                selected_times = np.full(ts_len, np.nan)
                selected_values = np.full(shape_values + (ts_len,), np.nan)

            else: 
                times, values = pad_timeseries_to_interval(
                    times,
                    values,
                    dt,
                    t_cut + self.delta, # t_start
                    t_cut + self.delta + self.output_length, # t_end
                    shape_values,
                )

                # idx = np.where(times >= t_cut + self.delta)[0][:ts_len]
                cut_time = t_cut + self.delta
                start_idx = np.searchsorted(times, cut_time, side="left")
                idx = np.arange(start_idx, start_idx + ts_len)
                idx = np.clip(idx, 0, len(times) - 1)

                # chosen = idx[:ts_len]
                # print('out', len(idx))

                selected_times = times[idx]
                selected_values = values[..., idx]
            
            # print(selected_values)
            
            if selected_values.ndim == 2 and selected_values.shape[0] == 1:
                selected_values = selected_values[0]

            out[key] = {"time": selected_times, "values": selected_values}

        return out

    # ------------------------------------------------------------------
    # ACTUATOR
    # ------------------------------------------------------------------
    def _build_actuator(self, sample, t_cut):

        out = {}

        for key in self.actuator_keys:
            md = self.dict_task_metadata["actuator"][key]
            freq = md["dt"]
            dt = md["dt"]
            shape_values = tuple(md["values_shape"])

            ts_in = int(round(self.input_length / freq))
            ts_out = int(round(self.output_length / freq))
            ts_delta = int(round(self.delta / freq))
            # print('ts_delta', ts_delta)

            times = sample[key]["time"]
            values = sample[key]["values"]

            if times.size==0 or values.size==0:
                selected_times = np.full(ts_in + ts_delta + ts_out, np.nan)
                selected_values = np.full(shape_values + (ts_in + ts_delta + ts_out,), np.nan)

            else: 
                times, values = pad_timeseries_to_interval(
                    times,
                    values,
                    dt,
                    t_cut - self.input_length, # t_start
                    t_cut + self.delta + self.output_length, # t_end
                    shape_values,
                )

                # idx_in = np.where(times < t_cut)[0][-ts_in:]
                # idx_out = np.where(times >= t_cut)[0][: ts_delta + ts_out]
                cut_idx = np.searchsorted(times, t_cut, side="left")
                idx_in = np.arange(cut_idx - ts_in, cut_idx)
                idx_in = np.clip(idx_in, 0, len(times) - 1)
                
                idx_out = np.arange(cut_idx, cut_idx + ts_delta + ts_out)
                idx_out = np.clip(idx_out, 0, len(times) - 1)

                # print('act in', len(idx_in), idx_in)
                # print('act out', len(idx_out), idx_out)

                selected_times = np.concatenate([times[idx_in], times[idx_out]])
                selected_values = np.concatenate([values[..., idx_in], values[..., idx_out]], axis=-1)
            
            # print(selected_values)

            if selected_values.ndim == 2 and selected_values.shape[0] == 1:
                selected_values = selected_values[0]

            out[key] = {"time": selected_times, "values": selected_values}

        return out

    # ------------------------------------------------------------------
    def get_shot_id(self, idx):
        return self.base.shots_list[idx]


    # ------------------------------------------------------------------
    # MARKOVIAN INPUT
    # ------------------------------------------------------------------
    def _build_non_markovian_input(self, sample, t_cut):

        out = {}

        for key in self.input_keys:
            md = self.dict_task_metadata["input"][key]
            ts_len = md["ts_length"]
            dt = md["dt"]
            shape_values = tuple(md["values_shape"])

            times = sample[key]["time"]
            values = sample[key]["values"]

            if times.size==0 or values.size==0:
                selected_times = np.full(ts_len, np.nan)
                selected_values = np.full(shape_values + (ts_len,), np.nan)

            else:
                times, values = pad_timeseries_to_interval(
                    times,
                    values,
                    dt,
                    t_cut - self.input_length, # t_start is the warmup period
                    t_cut, # t_end
                    shape_values,
                )

                # idx = np.where(times < t_cut)[0][-ts_len:]
                cut_idx = np.searchsorted(times, t_cut, side="left")
                idx = np.arange(0, cut_idx)
                # print('markovian in', len(idx))
                
                selected_times = times[idx]
                selected_values = values[..., idx]

            # print(selected_values)

            if selected_values.ndim == 2 and selected_values.shape[0] == 1:
                selected_values = selected_values[0]

            out[key] = {"time": selected_times, "values": selected_values}

        return out

    # ------------------------------------------------------------------
    # MARKOVIAN ACTUATOR
    # ------------------------------------------------------------------
    def _build_non_markovian_actuator(self, sample, t_cut):

        out = {}

        for key in self.actuator_keys:
            md = self.dict_task_metadata["actuator"][key]
            freq = md["dt"]
            dt = md["dt"]
            shape_values = tuple(md["values_shape"])

            ts_in = int(round(self.input_length / freq))
            ts_out = int(round(self.output_length / freq))
            ts_delta = int(round(self.delta / freq))
            # print('ts_delta', ts_delta)

            times = sample[key]["time"]
            values = sample[key]["values"]

            if times.size==0 or values.size==0:
                selected_times = np.full(ts_in + ts_delta + ts_out, np.nan)
                selected_values = np.full(shape_values + (ts_in + ts_delta + ts_out,), np.nan)

            else: 
                times, values = pad_timeseries_to_interval(
                    times,
                    values,
                    dt,
                    t_cut - self.input_length, # t_start is the warmup period
                    t_cut + self.delta + self.output_length, # t_end
                    shape_values,
                )

                # idx_in = np.where(times < t_cut)[0][-ts_in:]
                # idx_out = np.where(times >= t_cut)[0][: ts_delta + ts_out]
                cut_idx = np.searchsorted(times, t_cut, side="left")
                idx_in = np.arange(0, cut_idx)
                idx_in = np.clip(idx_in, 0, len(times) - 1)
                
                idx_out = np.arange(cut_idx, cut_idx + ts_delta + ts_out)
                idx_out = np.clip(idx_out, 0, len(times) - 1)

                # print('markovian act in', len(idx_in))
                # print('markovian act out', len(idx_out))

                selected_times = np.concatenate([times[idx_in], times[idx_out]])
                selected_values = np.concatenate([values[..., idx_in], values[..., idx_out]], axis=-1)
            
            # print(selected_values)

            if selected_values.ndim == 2 and selected_values.shape[0] == 1:
                selected_values = selected_values[0]

            out[key] = {"time": selected_times, "values": selected_values}

        return out


