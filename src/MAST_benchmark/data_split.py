import os

import pandas as pd
import random

from MAST_benchmark.tools.path import METADATA_DIR


# ----------------------------------------------------------------------------------------------------------------------
def get_train_test_val_shots(
    max_index = None,
    max_index_for_train = None,
    max_index_for_val = None,
    max_index_for_test = None,
    shuffle = False,
    seed = None
    ):
    
    """
    Generate lists of shot IDs for training, testing, and validation.
    These lists can be subsets of the corresponding complete lists.

    Parameters
    ----------
    max_index : int, optional
        If not None, all lists will have the same length given by max_index.
    max_index_for_train : int, optional
        Number of shot IDs for the training set.
        Overrides max_index.
    max_index_for_val : int, optional
        Number of shot IDs for the validation set.
        Overrides max_index.
    max_index_for_test : int, optional
        Number of shot IDs for the testing set.
        Overrides max_index.
    shuffle: bool
        True if we need shuffled samples.
    seed: int 
        For reproducibility of the rnd sequence.

    Returns
    -------
    tuple of lists
        Three lists of shot IDs for training, testing, and validation, respectively.

    """

    # Read full data splits
    file_path = os.path.join(METADATA_DIR, "data_splits.csv")
    train_set_full, test_set_full, val_set_full = read_data_split_csv(csv_path=file_path)

    if shuffle:
        if seed is not None:
            if not isinstance(seed, int):
                raise ValueError(f"Seed must be an integer, got {type(seed).__name__}")
            random.seed(seed)  
            
        random.shuffle(train_set_full)
        random.shuffle(test_set_full)
        random.shuffle(val_set_full)
        
    train_set = train_set_full
    test_set = test_set_full
    val_set = val_set_full
    
    # If max_index is provided, override all other limits
    if max_index is not None and max_index > 0:
        train_set = train_set_full[:max_index]
        val_set = val_set_full[:max_index]
        test_set = test_set_full[:max_index]

    # Apply individual limits if provided and positive
    if max_index_for_train is not None and max_index_for_train > 0:
        train_set = train_set_full[:max_index_for_train]

    if max_index_for_val is not None and max_index_for_val > 0:
        val_set = val_set_full[:max_index_for_val]

    if max_index_for_test is not None and max_index_for_test > 0:
        test_set = test_set_full[:max_index_for_test]
        
    return train_set, test_set, val_set


# ----------------------------------------------------------------------------------------------------------------------
def read_data_split_csv(csv_path):
    """Read the csv file containing the lists of shot IDs for
    training, validation and testing.
    """

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found at: {csv_path}")

    df = pd.read_csv(csv_path)

    shot_ids_for_train = df[df["train"] == True]["shot_id"].tolist()  # noqa
    shot_ids_for_test = df[df["test"] == True]["shot_id"].tolist()  # noqa
    shot_ids_for_val = df[df["val"] == True]["shot_id"].tolist()  # noqa

    return shot_ids_for_train, shot_ids_for_test, shot_ids_for_val
