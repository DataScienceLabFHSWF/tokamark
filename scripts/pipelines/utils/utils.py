
import os
import pickle
import joblib
import pandas as pd
import torch
import random
import numpy as np
import itertools
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import psutil, os

from globals import REPO_ROOT
from scripts.MAST_tools.MAST_dataset import MastDataset
from scripts.pipelines.preprocessing.sampled_shot_list import yamane_sampled_shot_list
from scripts.pipelines.preprocessing.standardscaling_preprocessing import get_mean_shot, get_std_shot


# ----------------------------------------------------------------------------------------------------------------------
def set_seed(seed: int, deterministic: bool = True, warn_only: bool = True):
    """
    Global reproducibility across Python, NumPy, and PyTorch (CPU/CUDA/MPS).
    Call once at startup, before building datasets/loaders/models.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    # Needed for strict cuBLAS determinism (matmul). Safe if CUDA isn't present.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # cuDNN + global deterministic guard
    try:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = bool(deterministic)
    except Exception as ee:
        print(f"WARNING - torch exception triggered: {ee}")
        pass

    torch.use_deterministic_algorithms(bool(deterministic), warn_only=bool(warn_only))


# ----------------------------------------------------------------------------------------------------------------------
def seed_worker(worker_id: int):
    """
    Top-level (picklable) worker init. Derives a per-worker seed from
    PyTorch's worker seed so it's consistent with DataLoader's Generator.
    """
    worker_seed = (torch.initial_seed() + worker_id) % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# ----------------------------------------------------------------------------------------------------------------------
def make_data_generator(seed: int) -> torch.Generator:
    """
    Top-level helper to create a reproducible DataLoader generator.
    """
    g = torch.Generator()
    g.manual_seed(seed)
    return g


# ----------------------------------------------------------------------------------------------------------------------
def dataloader_seed_parts(seed: int):
    # reuse the top-level function so it's picklable under 'spawn'
    return seed_worker, make_data_generator(seed)


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
    print(full_path)

    if not os.path.exists(full_path):
        raise FileNotFoundError(f"CSV not found at: {full_path}")

    df = pd.read_csv(full_path)

    shot_ids_for_train = df[df['train'] == True]['shot_id'].tolist()  # noqa
    shot_ids_for_test = df[df['test'] == True]['shot_id'].tolist()  # noqa
    shot_ids_for_val = df[df['val'] == True]['shot_id'].tolist()  # noqa

    return shot_ids_for_train, shot_ids_for_test, shot_ids_for_val


# ----------------------------------------------------------------------------------------------------------------------
def fit_mean_and_std_for_signal_transform(output_sub_dir, train_shots, source_signal_list, verbose=False, local=False):

    if verbose:
        print('\n\n----------TRANSFORM FITTING----------\n')

    preprocessing_train_dataset = MastDataset(
        local=local,
        shots_list=yamane_sampled_shot_list(train_shots, error=0.05),
        source_signal_list=source_signal_list,
        signal_level_transform_map=None,
        shot_level_transform=None
    )

    if verbose:
        print(f"len(preprocessing_train_dataset): {len(preprocessing_train_dataset)}")

    dict_mean_ = get_mean_shot(preprocessing_train_dataset)
    if verbose:
        print(f"dict_mean_ is {dict_mean_}")
    dict_std_ = get_std_shot(preprocessing_train_dataset)
    if verbose:
        print(f"dict_std_ is {dict_std_}")

    # Save dict_mean and dict_std used!

    output_dir_ = os.path.join("output", output_sub_dir)
    os.makedirs(output_dir_, exist_ok=True)
    if verbose:
        print(f"Output folder to save fitted mean and std dicts: {output_dir_}")

    with open(output_dir_ + 'dict_mean_shot.pkl', 'wb') as f_:
        pickle.dump(dict_mean_, f_)
    with open(output_dir_ + 'dict_std_shot.pkl', 'wb') as f_:
        pickle.dump(dict_std_, f_)

    return dict_mean_, dict_std_

# ----------------------------------------------------------------------------------------------------------------------
def initialize_datasets(
        sources_and_signals,
        shots,
        sig_tran_map,
        shot_tran,
        local_flag=False,
        return_incomplete_shots=True,
        verbose=False

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
        model_specific_transform,
        verbose=False

):
    datasets_ = {"train": None, "val": None, "test": None}

    # ..................................................................................................................
    # Train

    datasets_["train"] = TaskModelTransformWrapper(
                            datasets_train_val_test["train"],
                            dict_metadata,
                            config_task,
                            model_specific_transform,
                            verbose
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
                            verbose
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
                            verbose
                            )
    if verbose:
        print(f"len(mast_test_dataset): {len(datasets_['test'])}")

    # ..................................................................................................................
    # Return

    return datasets_


# ----------------------------------------------------------------------------------------------------------------------
def initialize_dataloaders(
        datasets,
        collate_function,
        batch_size,
        num_workers,
        shuffle=True,
        drop_last=False,
        verbose=False,
        seed: int | None = None,
        pin_memory: bool | None = None,
):
    dataloaders_ = {"train": None, "val": None, "test": None}

    if verbose:
        print('\n\n----------DATASET & DATALOADER INITIALIZATION----------\n')

    # ▶ Prepare reproducible seeding parts for DataLoader
    worker_fn = None
    generator = None
    if seed is not None:
        worker_fn = seed_worker
        generator = make_data_generator(seed)

    # sensible default for pin_memory
    if pin_memory is None:
        pin_memory = torch.cuda.is_available()

    # ..................................................................................................................
    # Train
    if datasets["train"]:
        dataloaders_["train"] = DataLoader(
            dataset=datasets["train"],
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=shuffle,
            drop_last=drop_last,
            collate_fn=collate_function,
            worker_init_fn=worker_fn,  # ▶
            generator=generator,  # ▶ controls shuffle order deterministically
            pin_memory=pin_memory,
        )

    # ..................................................................................................................
    # Val
    if datasets["val"]:
        dataloaders_["val"] = DataLoader(
            dataset=datasets["val"],
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=shuffle,
            drop_last=drop_last,
            collate_fn=collate_function,
            worker_init_fn=worker_fn,  # ▶ ensures worker RNG is fixed
            generator=generator,  # ▶ reproducible order if shuffle=True
            pin_memory=pin_memory,
        )

    # ..................................................................................................................
    # Test
    if datasets["test"]:
        dataloaders_["test"] = DataLoader(
            dataset=datasets["test"],
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=shuffle,
            drop_last=drop_last,
            collate_fn=collate_function,
            worker_init_fn=worker_fn,
            generator=generator,
            pin_memory=pin_memory,
        )

    return dataloaders_

# ======================================================================================================================
class TaskModelTransformWrapper(MastDataset):

    def __init__(self, base_dataset, dict_metadata, config_task, 
                 model_transform, 
                 verbose = False):
        
        self.base = base_dataset
        self.shots_list = self.base.shots_list
        self.dict_metadata = dict_metadata

        self.input_keys = [
            f"{source}-{signal}" for source, signal in (config_task["task_window_segmenter"]["input_keys"] or [])
        ] 
        self.actuator_keys = [
            f"{source}-{signal}" for source, signal in (config_task["task_window_segmenter"]["actuator_keys"] or [])
        ]
        self.output_keys = [
            f"{source}-{signal}" for source, signal in (config_task["task_window_segmenter"]["output_keys"] or [])
        ]

        self.input_length = config_task["task_window_segmenter"]["input_length"]
        self.output_length = config_task["task_window_segmenter"]["output_length"]
        self.delta = config_task["task_window_segmenter"]["delta"]
        
        self.model_transform = model_transform

        self.verbose = verbose

        if self.verbose:

            print(f"\nINPUT VARIABLES L={self.input_length}s")
            for key in self.input_keys:
                print(f"Variable {key}")
                freq_key = self.dict_metadata[key]["dt"]
                print(f"    frequency: {freq_key}")
                dim_key = self.dict_metadata[key]["values_shape"]
                print(f"    dim shape: {dim_key}")
                ts_length = np.trunc(self.input_length / freq_key)
                print(f"    we expect a window of length: {ts_length}")
            
            print(f"\nACTUATOR VARIABLES L={self.input_length+self.delta+self.output_length}s")
            for key in self.actuator_keys:
                print(f"Variable {key}")
                freq_key = self.dict_metadata[key]["dt"]
                print(f"    frequency: {freq_key}")
                dim_key = self.dict_metadata[key]["values_shape"]
                print(f"    dim shape: {dim_key}")
                ts_length = (np.trunc(self.input_length / freq_key) 
                             + np.trunc(self.output_length / freq_key) 
                             + np.trunc(self.delta / freq_key) )
                print(f"    we expect a window of length: { ts_length }")
            
            print(f"\nOUTPUT VARIABLES L={self.output_length}s")
            for key in self.output_keys:
                print(f"Variable {key}")
                freq_key = self.dict_metadata[key]["dt"]
                print(f"    frequency: {freq_key}")
                dim_key = self.dict_metadata[key]["values_shape"]
                print(f"    dim shape: {dim_key}")
                ts_length = np.trunc(self.output_length / freq_key)
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
        for var in self.output_keys: # only check output variables to determine if sample able to be run on!
            t = sample[var]["time"]
            if t.size != 0:
                t_start.append(t[0])
                t_end.append(t[-1])
                dts = np.diff(t)
                if len(dts) > 0:
                    delta_ts.append(np.min(dts))
                    # if self.verbose:
                    #     print(f"\nt_start for {var}: {t[0]:.6f}")
                    #     print(f"t_end for {var}: {t[-1]:.6f} s")
                    #     print(f"Δt for {var}: {np.min(dts):.6f} s")
        if not delta_ts:
            print(f"[Warning] No valid Δt found in any signals for shot {self.get_shot_id(idx_shot)}")  
            # yield {
            #     "shot_id": self.get_shot_id(idx_shot),
            #     "window_index": np.nan,
            #     "x": {np.nan},
            #     "y": {np.nan},
            # }
            return  # stop processing this sample
            
        start_time = np.min(t_start)
        end_time = np.max(t_end)

        max_dt = np.max(delta_ts)
        stride = max_dt # we could do min_dt

        t_cuts = np.arange(start_time, end_time, stride)
        # print(t_cuts)
        
        for idx_t, t_cut in enumerate(t_cuts):
            
            # ..........................................................................................................
            # Input 
            
            input_slice = {}

            for key in self.input_keys:

                freq_key = self.dict_metadata[key]["dt"]
                shape_values = self.dict_metadata[key]["values_shape"]
                ts_input = np.trunc(self.input_length / freq_key).astype(int)
                
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
                selected_times = np.concatenate([
                                            np.full(pad_len_input, np.nan),
                                            times[chosen_idx_in]
                                            ])
                
                # 4. Get padded values and handle when values empty
                if chosen_idx_in.size > 0:
                    sel_values_in = values[..., chosen_idx_in]
                else:
                    sel_values_in = np.empty(shape_values + (0,), dtype=values.dtype)
                selected_values = np.concatenate([
                                            np.full(shape_values + (pad_len_input,), np.nan, dtype=float), 
                                            sel_values_in,
                                            ], axis=-1)
                # 5. Get slice
                input_slice[key] = {
                    "time": selected_times,
                    "values": selected_values
                }
            
            # ..........................................................................................................
            # Actuator

            actuator_slice = {}

            for key in self.actuator_keys:

                freq_key = self.dict_metadata[key]["dt"]
                shape_values = self.dict_metadata[key]["values_shape"]
                ts_input = np.trunc(self.input_length / freq_key).astype(int)
                ts_output = np.trunc(self.output_length / freq_key).astype(int)
                ts_delta = np.trunc(self.delta / freq_key).astype(int)
                
                times = sample[key]["time"]
                values = sample[key]["values"]

                # 1. mask for times before t_end
                idx_in = np.where(times <= t_cut)[0]
                idx_out = np.where(times > t_cut+self.delta)[0]

                # 2. choose exactly ts_input indices (or fewer if not available)
                if len(idx_in) >= ts_input:
                    chosen_idx_in = idx_in[-ts_input:]
                else:
                    chosen_idx_in = idx_in
                
                if len(idx_out) >= ts_delta + ts_output :
                    chosen_idx_out = idx_out[:ts_delta+ts_output]
                else:
                    chosen_idx_out = idx_out

                # 3. Get padded time
                pad_len_input = ts_input - len(chosen_idx_in)
                pad_len_output = ts_delta + ts_output - len(chosen_idx_out)
                selected_times = np.concatenate([
                                            np.full(pad_len_input, np.nan),
                                            times[chosen_idx_in],
                                            times[chosen_idx_out],
                                            np.full(pad_len_output, np.nan),
                                            ])
                
                # 4. Get padded values and handle when values empty
                if chosen_idx_in.size > 0:
                    sel_values_in = values[..., chosen_idx_in]
                else:
                    sel_values_in = np.empty(shape_values + (0,), dtype=values.dtype)
                
                if chosen_idx_out.size > 0:
                    sel_values_out = values[..., chosen_idx_out]
                else:
                    sel_values_out = np.empty(shape_values + (0,), dtype=values.dtype)
                
                selected_values = np.concatenate([
                                            np.full(shape_values + (pad_len_input,), np.nan, dtype=float), 
                                            sel_values_in,
                                            sel_values_out,
                                            np.full(shape_values + (pad_len_output,), np.nan, dtype=float),
                                            ], axis=-1)
                
                # 5. Get slice
                actuator_slice[key] = {
                    "time": selected_times,
                    "values": selected_values
                }

            # ..........................................................................................................
            # Output
            
            output_slice = {}

            for key in self.output_keys:

                freq_key = self.dict_metadata[key]["dt"]
                shape_values = self.dict_metadata[key]["values_shape"]
                ts_output = np.trunc(self.output_length / freq_key).astype(int)
                ts_delta = np.trunc(self.delta / freq_key).astype(int)
                
                times = sample[key]["time"]
                values = sample[key]["values"]

                # 1. mask for times before t_end
                idx_out = np.where(times > t_cut+self.delta)[0]

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

                selected_times = np.concatenate([
                                            times[chosen_idx_out],
                                            np.full(pad_len_output, np.nan),
                                            ])
                
                # 4. Get padded values and handle when values empty
                if chosen_idx_out.size > 0:
                    sel_values_out = values[..., chosen_idx_out]
                else:
                    sel_values_out = np.empty(shape_values + (0,), dtype=values.dtype)

                selected_values = np.concatenate([
                                            sel_values_out,
                                            np.full(shape_values + (pad_len_output,), np.nan, dtype=float),
                                            ], axis=-1)
                
                # 5. Get slice
                output_slice[key] = {
                    "time": selected_times,
                    "values": selected_values
                }


            obj = {
                "input": input_slice,
                "actuator": actuator_slice,
                "output": output_slice
            }

            yield {
                "shot_id": self.get_shot_id(idx_shot),
                "window_index": idx_t,
                **self.model_transform(obj)
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


# ----------------------------------------------------------------------------------------------------------------------
def load_models(data_names, data_dir):
    """Load the PCA and imputer models for the given data names.

    Parameters
    ----------
    data_names : list[str]
        List of data names to load models for.
    data_dir : str
        Root data directory.

    Returns
    -------
    dict
        Dictionary containing the loaded PCA and imputer models.
    """

    pca_models = {}
    imputer_models = {}

    for source_name, signal_name in data_names:
        pca_model_path = f"{data_dir}pca_{signal_name}.joblib"
        imputer_model_path = f"{data_dir}imputer_{signal_name}.joblib"
        pca_models[f"{source_name}-{signal_name}"] = joblib.load(pca_model_path)
        imputer_models[f"{source_name}-{signal_name}"] = joblib.load(imputer_model_path)

    return {"pca": pca_models, "imputer": imputer_models}
