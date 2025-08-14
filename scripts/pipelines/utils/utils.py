import joblib
import os
import pandas as pd
from torch.utils.data._utils.collate import default_collate  # noqa
from typing import List, Tuple, Dict
import torch

# Compute project root relative to this file
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__) if '__file__' in globals()
                                         else os.getcwd(), "..", "..", ".."))
# print(REPO_ROOT)


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
def flatten_then_collate(batch):

    print(f"Collating batch of size {len(batch)}")
    
    # Flatten the batch of lists into a single list

    flattened_batch = None
    if isinstance(batch[0], list):
        flattened_batch = [item for sublist in batch for item in sublist]
        print(f'Number of samples from batch = {len(batch)} shots is N = {len(flattened_batch)}')

    # Use the default collate function
    return default_collate(flattened_batch) if (len(flattened_batch) > 0) else None


# ----------------------------------------------------------------------------------------------------------------------
def collate_fttransform(
    batch: List[Tuple[List, List[str], Dict[str, torch.Tensor]]],
    dtype: torch.dtype = torch.float32
):
    """
    collate_fttransform.py

    Custom collate function for FTTransformer-prepared samples.
    Intended to be used after flatten_then_collate or directly in DataLoader.

    Batch Format
    ------------
    Inputs:  batch = list of (Xs, names, y_native) tuples from FTTransformPrep.

    Outputs:
        X_batch       : List[B] of List[n_inputs] of np.ndarray
                        (model handles np→torch conversion internally)
        active_targets: List[str] target names shared by all samples in batch
        y_native      : Dict[target_name] -> (B, d1, d2, d3) torch tensor
    """
    if not batch:
        return [], [], {}

    # Ensure all samples have the same active_targets
    active_targets = batch[0][1]
    for _, names, _ in batch:
        if names != active_targets:
            raise ValueError("Batch mixes different active_targets; bucket by task before batching.")

    # Collect Xs
    X_batch = [sample[0] for sample in batch]

    # Stack Ys per target
    stacked: Dict[str, List[torch.Tensor]] = {n: [] for n in active_targets}
    for _, _, y_nat in batch:
        for n in active_targets:
            arr = torch.as_tensor(y_nat[n], dtype=dtype)
            stacked[n].append(arr)

    y_native = {n: torch.stack(tensors, dim=0).to(dtype) for n, tensors in stacked.items()}

    return X_batch, active_targets, y_native

# ======================================================================================================================
class ComposeTransforms(object):
    """Compose transforms and apply them in series checking for None return values

    Parameters
    ----------
    transforms : list[callable[tuple]]
        List containing the names of the transforms
    """

    # ----------------------------------------------------------------------------------------------------------------------
    def __init__(self, transforms):
        self.transforms = transforms

    # ----------------------------------------------------------------------------------------------------------------------
    def __call__(self, sample):
        for transform in self.transforms:
            if sample is None:
                return None
            sample = transform(sample)
        return sample

# ======================================================================================================================
def load_models(data_names, data_dir):
    """Load the PCA and imputer models for the given data names.

    Parameters
    ----------
    data_names : list[str]
        List of data names to load models for.

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