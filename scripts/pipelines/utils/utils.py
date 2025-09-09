import os
import sys
import csv

import pickle
import torch
import torch.multiprocessing as mp
from torch.utils.data import DataLoader

import joblib
import pandas as pd
from torch.utils.data._utils.collate import default_collate 
# from typing import List, Tuple, Dict
import torch

# Compute project root relative to this file
REPO_ROOT = os.path.abspath(os.path.join(
    os.path.dirname(__file__) if '__file__' in globals() else os.getcwd(),
    "..", "..", ".."
))  # noqa: E402

# print(REPO_ROOT)

from scripts.MAST_tools.MAST_dataset import MastDataset
from scripts.pipelines.preprocessing.sampled_shot_list import yamane_sampled_shot_list
from scripts.pipelines.preprocessing.standardscaling_preprocessing import get_mean_shot, get_std_shot


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
def get_train_test_val_shots(max_index=None):
    train_sh, test_sh, val_sh = read_data_split_csv()

    if max_index:
        train_sh = train_sh[0:max_index]
        val_sh = val_sh[0:max_index]
        test_sh = test_sh[0:max_index]

    return train_sh, test_sh, val_sh


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
            shot_level_transform=shot_tran
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
            shot_level_transform=shot_tran
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
            shot_level_transform=shot_tran
        )
        if verbose:
            print(f"len(test_dataset): {len(datasets_['test'])}")

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
        verbose=False
):

    dataloaders_ = {"train": None, "val": None, "test": None}

    if verbose:
        print('\n\n----------DATASET & DATALOADER INITIALIZATION----------\n')

    # ..................................................................................................................
    # Train

    if datasets["train"]:
        dataloaders_["train"] = DataLoader(
            dataset=datasets["train"],
            batch_size=batch_size,
            # batch_size=len(datasets['train']),
            num_workers=num_workers,
            shuffle=shuffle,
            drop_last=drop_last,
            collate_fn=collate_function
        )

    # ..................................................................................................................
    # Val

    if datasets["val"]:
        dataloaders_["val"] = DataLoader(
            dataset=datasets["val"],
            batch_size=batch_size,
            # batch_size=len(datasets["val"]),
            num_workers=num_workers,
            shuffle=shuffle,
            drop_last=drop_last,
            collate_fn=collate_function
        )

    # ..................................................................................................................
    # Test

    if datasets["test"]:
        dataloaders_["test"] = DataLoader(
            dataset=datasets["test"],
            batch_size=batch_size,
            # batch_size=len(datasets["test"]),
            num_workers=num_workers,
            shuffle=shuffle,
            drop_last=drop_last,
            collate_fn=collate_function
        )

    # ..................................................................................................................
    # Return

    return dataloaders_


# ----------------------------------------------------------------------------------------------------------------------
def collate_fttransform(batch, dtype: torch.dtype = torch.float32): # TO MOVE TOBIA
    """
    Custom collate for FTTransformer-prepared data.

    Parameters
    ----------
    batch : list
        Each element is a tuple:
          (Xs, active_targets, y_native)
          - Xs: list of np.ndarray (n_inputs)
          - active_targets: list of str
          - y_native: dict {target_name: np.ndarray (d1,d2,d3)}
    dtype : torch.dtype
        Target tensor dtype.

    Returns
    -------
    X_batch       : list of list[np.ndarray]  # (B, n_inputs)
    active_targets: list[str]  # shared for the whole batch
    y_native      : dict[target_name, torch.Tensor]  # (B, d1, d2, d3)
    """

    if not batch:
        return [], [], {}

    # Flatten if this is [ [sample, sample, ...], [sample, ...], ... ]
    if isinstance(batch[0], list):
        flat = []
        for sub in batch:
            flat.extend(sub)
        batch = flat

    # Check targets consistency
    active_targets = batch[0][1]
    for _, names, _ in batch:
        if names != active_targets:
            raise ValueError("Mixed active_targets in batch — bucket by task before batching.")

    # Collect inputs and outputs
    X_batch = [sample[0] for sample in batch]
    stacked = {t: [] for t in active_targets}

    for _, _, y_nat in batch:
        for t in active_targets:
            stacked[t].append(torch.as_tensor(y_nat[t], dtype=dtype))

    y_native = {t: torch.stack(vals, dim=0) for t, vals in stacked.items()}
    return X_batch, active_targets, y_native


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
