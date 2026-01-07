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
