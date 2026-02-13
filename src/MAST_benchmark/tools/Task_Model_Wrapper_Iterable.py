import numpy as np
import torch
from torch.utils.data import IterableDataset, get_worker_info
from typing import Optional, Mapping, Any


# --------------------------------------------------------------------------------------
# helpers 
# --------------------------------------------------------------------------------------

def all_vars_have_nans(d):
    for v in d.values():
        if not np.all(np.isnan(v["values"])):
            return False
    return True


def any_vars_have_nans(d):
    for v in d.values():
        if np.any(np.isnan(v["values"])):
            return True
    return False


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

            input_slice = self._build_input(sample, t_cut)
            output_slice = self._build_output(sample, t_cut)
            actuator_slice = self._build_actuator(sample, t_cut)

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
                        all_vars_have_nans(obj["input"])
                        and all_vars_have_nans(obj["actuator"])
                    )
                    and not any_vars_have_nans(obj["output"])
                )
            else:
                window_valid = True

            if not window_valid:
                continue

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
            shape_values = tuple(md["values_shape"])

            times = sample[key]["time"]
            values = sample[key]["values"]

            idx = np.where(times <= t_cut)[0]
            chosen = idx[-ts_len:]

            pad = ts_len - len(chosen)

            selected_times = np.concatenate(
                [np.full(pad, np.nan), times[chosen]]
            )

            if chosen.size:
                sel_values = values[..., chosen]
            else:
                sel_values = np.empty(shape_values + (0,))

            selected_values = np.concatenate(
                [
                    np.full(shape_values + (pad,), np.nan),
                    sel_values,
                ],
                axis=-1,
            )

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
            shape_values = tuple(md["values_shape"])

            times = sample[key]["time"]
            values = sample[key]["values"]

            idx = np.where(times > t_cut + self.delta)[0]
            chosen = idx[:ts_len]

            pad = ts_len - len(chosen)

            if chosen.size:
                sel_values = values[..., chosen]
            else:
                sel_values = np.empty(shape_values + (0,))

            selected_times = np.concatenate(
                [times[chosen], np.full(pad, np.nan)]
            )

            selected_values = np.concatenate(
                [
                    sel_values,
                    np.full(shape_values + (pad,), np.nan),
                ],
                axis=-1,
            )

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
            shape_values = tuple(md["values_shape"])

            ts_in = int(round(self.input_length / freq))
            ts_out = int(round(self.output_length / freq))
            ts_delta = int(round(self.delta / freq))

            times = sample[key]["time"]
            values = sample[key]["values"]

            idx_in = np.where(times <= t_cut)[0][-ts_in:]
            idx_out = np.where(times > t_cut + self.delta)[0][: ts_delta + ts_out]

            pad_in = ts_in - len(idx_in)
            pad_out = ts_delta + ts_out - len(idx_out)

            selected_times = np.concatenate([
                np.full(pad_in, np.nan),
                times[idx_in],
                times[idx_out],
                np.full(pad_out, np.nan),
            ])

            sel_in = values[..., idx_in] if idx_in.size else np.empty(shape_values + (0,))
            sel_out = values[..., idx_out] if idx_out.size else np.empty(shape_values + (0,))

            selected_values = np.concatenate([
                np.full(shape_values + (pad_in,), np.nan),
                sel_in,
                sel_out,
                np.full(shape_values + (pad_out,), np.nan),
            ], axis=-1)

            if selected_values.ndim == 2 and selected_values.shape[0] == 1:
                selected_values = selected_values[0]

            out[key] = {"time": selected_times, "values": selected_values}

        return out

    # ------------------------------------------------------------------
    def get_shot_id(self, idx):
        return self.base.shots_list[idx]
