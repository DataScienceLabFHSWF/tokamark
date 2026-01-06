import os
import pandas as pd
import numpy as np

from ..globals import REPO_ROOT
from scripts.MAST_tools.MAST_dataset import MastDataset


# ----------------------------------------------------------------------------------------------------------------------
def get_train_test_val_shots(max_index=None):
    train_sh, test_sh, val_sh = read_data_split_csv()

    if max_index:
        train_sh = train_sh[0:max_index]
        val_sh = val_sh[0:max_index]
        test_sh = test_sh[0:max_index]

    return train_sh, test_sh, val_sh


# ----------------------------------------------------------------------------------------------------------------------
def read_data_split_csv(csv_path="metadata/2025-05-12/data_splits.csv"):
    """Read the csv file containing the lists of shot IDs for
    training, validation and testing.
    """

    full_path = os.path.join(REPO_ROOT, csv_path)

    if not os.path.exists(full_path):
        raise FileNotFoundError(f"CSV not found at: {full_path}")

    df = pd.read_csv(full_path)

    shot_ids_for_train = df[df["train"] == True]["shot_id"].tolist()  # noqa
    shot_ids_for_test = df[df["test"] == True]["shot_id"].tolist()  # noqa
    shot_ids_for_val = df[df["val"] == True]["shot_id"].tolist()  # noqa

    return shot_ids_for_train, shot_ids_for_test, shot_ids_for_val


# ----------------------------------------------------------------------------------------------------------------------
def initialize_datasets(
    sources_and_signals,
    shots,
    sig_tran_map,
    shot_tran,
    local_flag=False,
    return_incomplete_shots=True,
    verbose=False,
):
    datasets_ = {"train": None, "val": None, "test": None}

    # ..................................................................................................................
    # Train

    if shots["train"]:
        datasets_["train"] = MastDataset(
            local=local_flag,
            shots_list=shots["train"],
            source_signal_list=sources_and_signals,
            signal_level_transform_map=sig_tran_map,
            shot_level_transform=shot_tran,
            return_incomplete_shots=return_incomplete_shots,
        )
        if verbose:
            print(f"len(mast_train_dataset): {len(datasets_['train'])}")

    # ..................................................................................................................
    # Val

    if shots["val"]:
        datasets_["val"] = MastDataset(
            local=local_flag,
            shots_list=shots["val"],
            source_signal_list=sources_and_signals,
            signal_level_transform_map=sig_tran_map,
            shot_level_transform=shot_tran,
            return_incomplete_shots=return_incomplete_shots,
        )
        if verbose:
            print(f"len(val_dataset): {len(datasets_['val'])}")

    # ..................................................................................................................
    # Test

    if shots["test"]:
        datasets_["test"] = MastDataset(
            local=local_flag,
            shots_list=shots["test"],
            source_signal_list=sources_and_signals,
            signal_level_transform_map=sig_tran_map,
            shot_level_transform=shot_tran,
            return_incomplete_shots=return_incomplete_shots,
        )
        if verbose:
            print(f"len(test_dataset): {len(datasets_['test'])}")

    # ..................................................................................................................
    # Return

    return datasets_


# ----------------------------------------------------------------------------------------------------------------------
def initialize_model_datasets(
    datasets_train_val_test,
    dict_metadata,
    config_task,
    model_specific_transform=None,
    verbose=False,
):
    datasets_ = {"train": None, "val": None, "test": None}

    # ..................................................................................................................
    # Train

    datasets_["train"] = TaskModelTransformWrapper(
        datasets_train_val_test["train"],
        dict_metadata,
        config_task,
        model_specific_transform,
        verbose,
    )
    if verbose:
        print(f"len(mast_train_dataset): {len(datasets_['train'])}")

    # ..................................................................................................................
    # Val

    datasets_["val"] = TaskModelTransformWrapper(
        datasets_train_val_test["val"],
        dict_metadata,
        config_task,
        model_specific_transform,
        verbose=False,
    )
    if verbose:
        print(f"len(mast_val_dataset): {len(datasets_['val'])}")

    # ..................................................................................................................
    # Test

    datasets_["test"] = TaskModelTransformWrapper(
        datasets_train_val_test["test"],
        dict_metadata,
        config_task,
        model_specific_transform,
        verbose=False,
    )
    if verbose:
        print(f"len(mast_test_dataset): {len(datasets_['test'])}")

    # ..................................................................................................................
    # Return

    return datasets_


# ======================================================================================================================
class TaskModelTransformWrapper(MastDataset):
    def __init__(
        self,
        base_dataset,
        dict_metadata,
        config_task,
        model_transform=None,
        verbose=False,
    ):
        self.base = base_dataset
        self.shots_list = self.base.shots_list
        self.dict_metadata = dict_metadata

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

        self.stride = float(self.dict_metadata["sec_stride"])

        self.model_transform = model_transform

        self.verbose = verbose

        if self.verbose:
            print(f"\nINPUT VARIABLES L={self.input_length}s")
            for key in self.input_keys:
                md = self.dict_metadata["input"][key]
                freq_key = md["dt"]
                dim_key = md["values_shape"]
                ts_length = md["ts_length"]
                print(f"Variable {key}")
                print(f"    frequency: {freq_key}")
                print(f"    dim shape: {dim_key}")
                print(f"    we expect a window of length: {ts_length}")

            print(
                f"\nACTUATOR VARIABLES L={self.input_length + self.delta + self.output_length}s"
            )
            for key in self.actuator_keys:
                md = self.dict_metadata["actuator"][key]
                freq_key = md["dt"]
                dim_key = md["values_shape"]
                ts_length = md["ts_length"]
                print(f"Variable {key}")
                print(f"    frequency: {freq_key}")
                print(f"    dim shape: {dim_key}")
                print(f"    we expect a window of length: {ts_length}")

            print(f"\nOUTPUT VARIABLES L={self.output_length}s")
            for key in self.output_keys:
                md = self.dict_metadata["output"][key]
                freq_key = md["dt"]
                dim_key = md["values_shape"]
                ts_length = md["ts_length"]
                print(f"Variable {key}")
                print(f"    frequency: {freq_key}")
                print(f"    dim shape: {dim_key}")
                print(f"    we expect a window of length: {ts_length}")

    def __getitem__(self, idx_shot):
        # process = psutil.Process(os.getpid())
        # mem = process.memory_info().rss / (1024 ** 2)
        # print(f"[Worker {os.getpid()}] __getitem__ mem: {mem:.2f} MB, idx={self.shots_list[idx]}")

        sample = self.base[idx_shot]

        t_start = []
        t_end = []
        delta_ts = []
        # for var in self.input_keys + self.actuator_keys + self.output_keys:
        for var in (
            self.output_keys
        ):  # only check output variables to determine if sample able to be run on!
            t = sample[var]["time"]
            if t.size != 0:
                t_start.append(t[0])
                t_end.append(t[-1])
                dts = np.diff(t)
                if len(dts) > 0:
                    delta_ts.append(np.min(dts))
        if not delta_ts:
            print(
                f"[Warning] No valid Δt found in any output signals for shot {self.get_shot_id(idx_shot)}"
            )
            return  # stop processing this sample

        start_time = np.min(t_start)
        end_time = np.max(t_end)

        t_cuts = np.arange(start_time, end_time, self.stride)
        # print(t_cuts)

        for idx_t, t_cut in enumerate(t_cuts):
            # ..........................................................................................................
            # Input

            input_slice = {}

            for key in self.input_keys:
                md = self.dict_metadata["input"][key]
                freq_key = md["dt"]
                shape_values = md["values_shape"]
                ts_input = md["ts_length"]

                times = sample[key]["time"]
                values = sample[key]["values"]

                # 1. mask for times before t_end
                idx_in = np.where(times <= t_cut)[0]

                # 2. choose exactly ts_input indices (or fewer if not available)
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
                md = self.dict_metadata["output"][key]
                freq_key = md["dt"]
                shape_values = md["values_shape"]
                ts_output = md["ts_length"]

                ts_delta = np.trunc(self.delta / freq_key).astype(int)

                times = sample[key]["time"]
                values = sample[key]["values"]

                # 1. mask for times before t_end
                idx_out = np.where(times > t_cut + self.delta)[0]

                # 2. choose exactly ts_input indices (or fewer if not available)
                if len(idx_out) >= ts_output:
                    chosen_idx_out = idx_out[:ts_output]
                else:
                    chosen_idx_out = idx_out

                # 3. Get padded time
                pad_len_output = ts_output - len(chosen_idx_out)

                if chosen_idx_out.size > 0:
                    sel_values_out = values[..., chosen_idx_out]
                else:
                    sel_values_out = np.empty(shape_values + (0,), dtype=values.dtype)

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
                md = self.dict_metadata["actuator"][key]
                freq_key = md["dt"]
                shape_values = md["values_shape"]

                ts_input = int(np.round(self.input_length / freq_key))
                ts_output = int(np.round(self.output_length / freq_key))
                ts_delta = int(np.round(self.delta / freq_key))

                times = sample[key]["time"]
                values = sample[key]["values"]

                # 1. mask for times before t_end
                idx_in = np.where(times <= t_cut)[0]
                idx_out = np.where(times > t_cut + self.delta)[0]

                # 2. choose exactly ts_input indices (or fewer if not available)
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
                "shot_id": self.get_shot_id(
                    idx_shot
                ),  # we need this to cache shots for external models
                "window_index": idx_t,  # we need this to cache shots for external models
            }

            obj2 = self.model_transform(obj) if self.model_transform else obj
            if obj2 is None:
                continue
            yield {
                "shot_id": self.get_shot_id(idx_shot),
                "window_index": idx_t,
                **obj2,
            }

    def __len__(self):
        return len(self.base)

    def get_shot_id(self, idx: int):
        return self.base.shots_list[idx]


# ======================================================================================================================
class ComposeTransforms(object):
    """Compose transforms and apply them in series checking for None return values

    Parameters
    ----------
    transforms : list[callable[tuple]]
        List containing the names of the transforms
    """

    # ------------------------------------------------------------------------------------------------------------------
    def __init__(self, transforms):
        self.transforms = transforms

    # ------------------------------------------------------------------------------------------------------------------
    def __call__(self, sample):
        for transform in self.transforms:
            if sample is None:
                return None
            sample = transform(sample)
        return sample
