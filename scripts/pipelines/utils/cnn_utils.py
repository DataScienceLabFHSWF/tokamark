import os
import sys
import csv

import pickle
import torch
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
from multiprocessing import cpu_count


# ----------------------------------------------------------------------------------------------------------------------
# Repo-specific imports

# Add the repo root (e.g.,/fairmast-data-preprocessing) to sys.path
REPO_ROOT = os.path.abspath(os.path.join(
    os.path.dirname(__file__) if '__file__' in globals() else os.getcwd(),
    "..", ".."
))  # noqa: E402

if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
# print(f"REPO_ROOT: {REPO_ROOT}")

from scripts.MAST_tools.MAST_dataset import MastDataset
from scripts.pipelines.utils.utils import read_data_split_csv, flatten_then_collate
from scripts.pipelines.preprocessing.sampled_shot_list import yamane_sampled_shot_list
from scripts.pipelines.preprocessing.standardscaling_preprocessing import get_mean_shot, get_std_shot
from scripts.pipelines.utils.utils import ComposeTransforms

from scripts.pipelines.transforms.signal_level_transforms.pretrained_stdscale_normalize_transform import (
    StdScalingTransform
)
from scripts.pipelines.transforms.signal_level_transforms.sampling_reference_time_transform import (
    SamplingToReferenceTimeTransform
)
from scripts.pipelines.transforms.shot_level_transforms.truncation_transform import (
    TruncationTransform
)
from scripts.pipelines.transforms.shot_level_transforms.window_segmenter_transform import (
    WindowSegmenterTransform
)
from scripts.pipelines.transforms.signal_level_transforms.fill_profile_with_zeros_imputer_transform import (
    FillProfileWithZerosTransform
)
from scripts.pipelines.transforms.shot_level_transforms.drop_sample_with_nans import (
    DropSampleWithNans
)
from scripts.pipelines.transforms.shot_level_transforms.cnn_transform import CNNTransform

from scripts.pipelines.models.cnn_model import MultiBranchCNNModel


# ----------------------------------------------------------------------------------------------------------------------
# Determine device to train on

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


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