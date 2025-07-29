import joblib
import os
import pandas as pd
from torch.utils.data._utils.collate import default_collate  # noqa

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

# ----------------------------------------------------------------------------------------------------------------------
def read_data_split_csv(csv_path="metadata/2025-05-12/data_splits.csv"):
    """Read the csv file containing the lists of shot IDs for 
    training, validation and testing.

    Parameters
    ----------
    csv_path : str, optional
        by default "metadata/2025-05-12/data_splits.csv"

    Returns
    -------
    Three lists containing shot IDs for training, validation and testing sets
    """
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"CSV file not found at: {csv_path}")

    # Filter rows where the 'train' column is True
    shot_ids_for_train = df[df['train'] == True]['shot_id'].tolist() 
    shot_ids_for_test = df[df['test'] == True]['shot_id'].tolist() 
    shot_ids_for_val = df[df['val'] == True]['shot_id'].tolist() 

    return shot_ids_for_train, shot_ids_for_test, shot_ids_for_val