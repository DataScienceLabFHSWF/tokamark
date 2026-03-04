"""
Docstring reference: https://numpydoc.readthedocs.io/en/latest/format.html
Python style reference: https://google.github.io/styleguide/pyguide.html
"""

import numpy as np
from typing import Optional, Any
from collections.abc import Mapping, Generator

from MAST_tools.MAST_dataset import MastDataset


# ----------------------------------------------------------------------------------------------------------------------
def all_vars_have_nans(
        dict_obj: Mapping[str, Any]
) -> bool:
    """
    Check if bool(x) is True for all values x in `dict_obj`.

    Parameters
    ----------
    dict_obj : Mapping[str, Any]
        Input dictionary.

    Returns
    -------
    bool
        If the iterable (dictionary) is empty, return True.

    """

    # print("ALL?", [np.isnan(np.asarray(dict_obj[var]["values"])).any() for var in dict_obj.keys()] )
    return all([np.isnan(np.asarray(dict_obj[var]["values"])).any() for var in dict_obj.keys()])


# ----------------------------------------------------------------------------------------------------------------------
def any_vars_have_nans(
        dict_obj: Mapping[str, Any]
) -> bool:
    """
    Check if bool(x) is True for any values x in `dict_obj`.

    Parameters
    ----------
    dict_obj : Mapping[str, Any]
        Input dictionary.

    Returns
    -------
    bool
        If the iterable (dictionary) is empty, return True.

    """

    # print("ANY?", [np.isnan(np.asarray(dict_obj[var]["values"])).any() for var in dict_obj.keys()] )
    return any([np.isnan(np.asarray(dict_obj[var]["values"])).any() for var in dict_obj.keys()])


# ======================================================================================================================
class TaskModelTransformWrapper(MastDataset):
    """
    Wrapper for task model transform.

    Attributes
    ----------
    base : MastDataset
        Target base MAST dataset.
    shots_list : tuple
        List of shots from the target base MAST dataset.
    dict_task_metadata : Mapping[str, Any]
        Metadata dictionary produced by the baseline pipeline (dt, shapes, etc.).
    input_keys : tuple
        List of input keys for the task window segmenter.
    actuator_keys : tuple
        List of actuator keys for the task window segmenter.
    output_keys : tuple
        List of output keys for the task window segmenter.
    input_length : int
        Input length for the task window segmenter.
    output_length : int
        Output length for the task window segmenter.
    delta : int
        Target delta for the task window segmenter.
    stride : float
        Target stride.
    model_transform : Any
        Target model transform.
    test_mode: bool
        If True, keeping window with any no nans input or actuator, and all full outputs.
    verbose : bool
        If True, activate verbose mode.

    Methods
    -------
    __getitem__(idx)
        Return dataset item by shot index.
    __len__()
        Get length of base dataset.
    get_shot_id(idx)
        Get shot ID from shot index.

    """

    # ------------------------------------------------------------------------------------------------------------------
    def __init__(
        self,
        base_dataset: MastDataset,
        dict_task_metadata: Mapping[str, Any],
        config_task: Mapping[str, Any],
        model_transform: Optional[Any] = None,
        test_mode: bool = False,
        verbose: bool = False,
    ) -> None:
        """
        Initialize class attributes.

        Parameters
        ----------
        base_dataset : Optional[MastDataset]
            Baseline shot-level dataset (e.g., MastDataset) for one split, or None.  # FIXME: How base_dataset is None?
        dict_task_metadata : Mapping[str, Any]
            Metadata dictionary produced by the baseline pipeline (dt, shapes, etc.).
        config_task : Mapping[str, Any]
            Task configuration dictionary containing `task_window_segmenter` (keys, lengths, delta).
        model_transform : Optional[Any]
            Optional model-specific transform chain applied per window.
            Optional. Default: None.
        test_mode : bool
            If True, keeping window with any no nans input or actuator, and all full outputs.
            Optional. Default: False.
        verbose : bool
            If True, activate verbose mode.
            Optional. Default: False.

        Returns
        -------
        None

        """

        super().__init__(  # TODO: Check with Cecile if this is the intended behavior.
            local=base_dataset.local,
            shots_list=base_dataset.shots_list,
            source_signal_list=base_dataset.source_signal_list
        )

        self.base = base_dataset
        self.shots_list = self.base.shots_list
        self.dict_task_metadata = dict_task_metadata

        self.input_keys = [
            f"{source}-{signal}"
            for source, signal in (
                config_task["task_window_segmenter"]["input_keys"] or []
            )
        ]
        self.actuator_keys = [
            f"{source}-{signal}"
            for source, signal in (
                config_task["task_window_segmenter"]["actuator_keys"] or []
            )
        ]
        self.output_keys = [
            f"{source}-{signal}"
            for source, signal in (
                config_task["task_window_segmenter"]["output_keys"] or []
            )
        ]

        self.input_length = config_task["task_window_segmenter"]["input_length"]
        self.output_length = config_task["task_window_segmenter"]["output_length"]
        self.delta = config_task["task_window_segmenter"]["delta"]

        self.stride = float(self.dict_task_metadata["sec_stride"])

        self.model_transform = model_transform
        self.test_mode = test_mode

        self.verbose = verbose

        if self.verbose:
            print(f"\nINPUT VARIABLES L={self.input_length}s")
            for key in self.input_keys:
                md = self.dict_task_metadata["input"][key]
                freq_key = md["dt"]
                dim_key = tuple(md["values_shape"])
                ts_length = md["ts_length"]
                print(f"Variable {key}")
                print(f"    frequency: {freq_key}")
                print(f"    dim shape: {dim_key}")
                print(f"    we expect a window of length: {ts_length}")

            print(
                f"\nACTUATOR VARIABLES L={self.input_length + self.delta + self.output_length}s"
            )
            for key in self.actuator_keys:
                md = self.dict_task_metadata["actuator"][key]
                freq_key = md["dt"]
                dim_key = tuple(md["values_shape"])
                ts_length = md["ts_length"]
                print(f"Variable {key}")
                print(f"    frequency: {freq_key}")
                print(f"    dim shape: {dim_key}")
                print(f"    we expect a window of length: {ts_length}")

            print(f"\nOUTPUT VARIABLES L={self.output_length}s")
            for key in self.output_keys:
                md = self.dict_task_metadata["output"][key]
                freq_key = md["dt"]
                dim_key = tuple(md["values_shape"])
                ts_length = md["ts_length"]
                print(f"Variable {key}")
                print(f"    frequency: {freq_key}")
                print(f"    dim shape: {dim_key}")
                print(f"    we expect a window of length: {ts_length}")

    # ------------------------------------------------------------------------------------------------------------------
    def __getitem__(
            self,
            shot_idx: int
    ) -> Generator[dict[str, Any], Any, None]:
        """
        Return dataset item by shot index.

        Parameters
        ----------
        shot_idx : int
            Shot index.

        Returns
        -------
        Generator[dict[str, Any], Any, None]
            Either yield dict[str, Any] with shot ID and window data for all valid windows, or return None if sample
            cannot be run on.

        """

        # process = psutil.Process(os.getpid())
        # mem = process.memory_info().rss / (1024 ** 2)
        # print(f"[Worker {os.getpid()}] __getitem__ mem: {mem:.2f} MB, idx={self.shots_list[idx]}")

        sample = self.base[shot_idx]

        t_start = []
        t_end = []
        delta_ts = []
        # for var in self.input_keys + self.actuator_keys + self.output_keys:  # TODO: Should we keep this line?
        for var in self.output_keys:  # Only check output variables to determine if sample can be run on!
            t = sample[var]["time"]
            if t.size != 0:
                t_start.append(t[0])
                t_end.append(t[-1])
                dts = np.diff(t)
                if len(dts) > 0:
                    delta_ts.append(np.min(dts))
        if not delta_ts:
            if self.verbose:
                print(
                    f"[Warning] No valid Δt found in any output signals for shot {self.get_shot_id(idx=shot_idx)}"
                )
            return  # Stop processing this sample

        start_time = np.min(t_start)
        end_time = np.max(t_end)

        t_cuts = np.arange(start_time, end_time, self.stride)
        # print(t_cuts)

        for idx_t, t_cut in enumerate(t_cuts):
            # ..........................................................................................................
            # Input

            input_slice = {}

            for key in self.input_keys:
                md = self.dict_task_metadata["input"][key]
                freq_key = md["dt"]                                                             # FIXME: Unused variable
                shape_values = tuple(md["values_shape"])
                ts_input = md["ts_length"]

                times = sample[key]["time"]
                values = sample[key]["values"]

                # 1. Mask for times before t_end
                idx_in = np.where(times <= t_cut)[0]

                # 2. Choose exactly ts_input indices (or fewer if not available)
                if len(idx_in) >= ts_input:
                    chosen_idx_in = idx_in[-ts_input:]
                else:
                    chosen_idx_in = idx_in

                # 3. Get padded time
                pad_len_input = ts_input - len(chosen_idx_in)
                selected_times = np.concatenate(
                    [np.full(pad_len_input, np.nan), times[chosen_idx_in]]
                )

                # 4. Get padded values and handle when values empty
                if chosen_idx_in.size > 0:
                    sel_values_in = values[..., chosen_idx_in]
                else:
                    sel_values_in = np.empty(shape_values + (0,), dtype=values.dtype)
                selected_values = np.concatenate(
                    [
                        np.full(shape_values + (pad_len_input,), np.nan, dtype=float),
                        sel_values_in,
                    ],
                    axis=-1,
                )

                # 5. Collapse if 2D time series
                if selected_values.ndim == 2 and selected_values.shape[0] == 1:
                    selected_values = selected_values[0]

                # 6. Get slice
                input_slice[key] = {"time": selected_times, "values": selected_values}

            # ..........................................................................................................
            # Output

            output_slice = {}

            for key in self.output_keys:
                md = self.dict_task_metadata["output"][key]
                freq_key = md["dt"]
                shape_values = tuple(md["values_shape"])
                ts_output = md["ts_length"]

                ts_delta = np.trunc(self.delta / freq_key).astype(int)                          # FIXME: Unused variable

                times = sample[key]["time"]
                values = sample[key]["values"]

                # 1. Mask for times before t_end
                idx_out = np.where(times > t_cut + self.delta)[0]

                # 2. Choose exactly ts_input indices (or fewer if not available)
                if len(idx_out) >= ts_output:
                    chosen_idx_out = idx_out[:ts_output]
                else:
                    chosen_idx_out = idx_out

                # 3. Get padded time
                pad_len_output = ts_output - len(chosen_idx_out)

                if chosen_idx_out.size > 0:
                    sel_values_out = values[..., chosen_idx_out]                                # FIXME: Unused variable
                else:
                    sel_values_out = np.empty(shape_values + (0,), dtype=values.dtype)          # FIXME: Unused variable

                selected_times = np.concatenate(
                    [
                        times[chosen_idx_out],
                        np.full(pad_len_output, np.nan),
                    ]
                )

                # 4. Get padded values and handle when values empty
                if chosen_idx_out.size > 0:
                    sel_values_out = values[..., chosen_idx_out]
                else:
                    sel_values_out = np.empty(shape_values + (0,), dtype=values.dtype)

                selected_values = np.concatenate(
                    [
                        sel_values_out,
                        np.full(shape_values + (pad_len_output,), np.nan, dtype=float),
                    ],
                    axis=-1,
                )

                # 5. Collapse if 2D time series
                if selected_values.ndim == 2 and selected_values.shape[0] == 1:
                    selected_values = selected_values[0]

                # 6. Get slice
                output_slice[key] = {"time": selected_times, "values": selected_values}

            # ..........................................................................................................
            # Actuator

            actuator_slice = {}

            for key in self.actuator_keys:
                md = self.dict_task_metadata["actuator"][key]
                freq_key = md["dt"]
                shape_values = tuple(md["values_shape"])

                ts_input = int(np.round(self.input_length / freq_key))
                ts_output = int(np.round(self.output_length / freq_key))
                ts_delta = int(np.round(self.delta / freq_key))

                times = sample[key]["time"]
                values = sample[key]["values"]

                # 1. Mask for times before t_end
                idx_in = np.where(times <= t_cut)[0]
                idx_out = np.where(times > t_cut + self.delta)[0]

                # 2. Choose exactly ts_input indices (or fewer if not available)
                if len(idx_in) >= ts_input:
                    chosen_idx_in = idx_in[-ts_input:]
                else:
                    chosen_idx_in = idx_in

                if len(idx_out) >= ts_delta + ts_output:
                    chosen_idx_out = idx_out[: ts_delta + ts_output]
                else:
                    chosen_idx_out = idx_out

                # 3. Get padded time
                pad_len_input = ts_input - len(chosen_idx_in)
                pad_len_output = ts_delta + ts_output - len(chosen_idx_out)
                selected_times = np.concatenate(
                    [
                        np.full(pad_len_input, np.nan),
                        times[chosen_idx_in],
                        times[chosen_idx_out],
                        np.full(pad_len_output, np.nan),
                    ]
                )

                # 4. Get padded values and handle when values empty
                if chosen_idx_in.size > 0:
                    sel_values_in = values[..., chosen_idx_in]
                else:
                    sel_values_in = np.empty(shape_values + (0,), dtype=values.dtype)

                if chosen_idx_out.size > 0:
                    sel_values_out = values[..., chosen_idx_out]
                else:
                    sel_values_out = np.empty(shape_values + (0,), dtype=values.dtype)

                selected_values = np.concatenate(
                    [
                        np.full(shape_values + (pad_len_input,), np.nan, dtype=float),
                        sel_values_in,
                        sel_values_out,
                        np.full(shape_values + (pad_len_output,), np.nan, dtype=float),
                    ],
                    axis=-1,
                )

                # 5. Collapse if 2D time series
                if selected_values.ndim == 2 and selected_values.shape[0] == 1:
                    selected_values = selected_values[0]

                # 6. Get slice
                actuator_slice[key] = {
                    "time": selected_times,
                    "values": selected_values,
                }

            obj = {
                "input": input_slice,
                "actuator": actuator_slice,
                "output": output_slice,
                "t_cut": t_cut,
                "shot_id": self.get_shot_id(idx=shot_idx),  # We need this to cache shots for external models
                "window_index": idx_t,  # We need this to cache shots for external models
            }

            if self.test_mode:
                window_valid = (
                    not (all_vars_have_nans(obj["input"]) and all_vars_have_nans(obj["actuator"]))
                    and not any_vars_have_nans(obj["output"])
                )
            else:
                window_valid = True

            if window_valid:
                obj2 = self.model_transform(obj) if self.model_transform else obj
                if obj2 is None:
                    continue
                yield {
                    "shot_id": self.get_shot_id(idx=shot_idx),
                    **obj2,
                }
            else:
                # If window not valid
                continue

    # ------------------------------------------------------------------------------------------------------------------
    def __len__(self) -> int:
        """Get length of base dataset."""

        return len(self.base)

    # ------------------------------------------------------------------------------------------------------------------
    def get_shot_id(
            self,
            idx: int
    ) -> int:
        """Get shot ID from shot index."""

        return self.base.shots_list[idx]
