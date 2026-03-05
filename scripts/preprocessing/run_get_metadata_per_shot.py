"""
Docstring reference: https://numpydoc.readthedocs.io/en/latest/format.html
Python style reference: https://google.github.io/styleguide/pyguide.html
"""

import os
import csv
import argparse
import yaml
import json
import numpy as np
from typing import Any
from multiprocessing import cpu_count
import torch.multiprocessing as mp
from torch.utils.data import Dataset, DataLoader

from MAST_benchmark.data import initialize_MAST_dataset
from MAST_benchmark.data_split import get_train_test_val_shots
from MAST_tools.MAST_dataset import MastDataset
from MAST_tools.constants import (
    DEFAULT_BASE_LOCAL_ZARR_PATH,
    DEFAULT_CONFIG_GET_METADATA_FILE,  # noqa
    DEFAULT_CONFIG_GET_METADATA_DEMO_FILE,  # noqa
    DEFAULT_SHOTS_STATS_VAL_FILE,
    DEFAULT_SHOTS_STATS_TEST_FILE,
    DEFAULT_SHOTS_STATS_TRAIN_FILE,
)

# ----------------------------------------------------------------------------------------------------------------------

DEMO_MODE = False  # TODO: Recalculate files using DEMO_MODE = False. [Rodrigo]

if DEMO_MODE:
    default_config = DEFAULT_CONFIG_GET_METADATA_DEMO_FILE
    default_shots_stats_val_file = DEFAULT_SHOTS_STATS_VAL_FILE.replace(".csv", "_DEMO.csv")
    default_shots_stats_test_file = DEFAULT_SHOTS_STATS_TEST_FILE.replace(".csv", "_DEMO.csv")
    default_shots_stats_train_file = DEFAULT_SHOTS_STATS_TRAIN_FILE.replace(".csv", "_DEMO.csv")
else:
    default_config = DEFAULT_CONFIG_GET_METADATA_FILE
    default_shots_stats_val_file = DEFAULT_SHOTS_STATS_VAL_FILE
    default_shots_stats_test_file = DEFAULT_SHOTS_STATS_TEST_FILE
    default_shots_stats_train_file = DEFAULT_SHOTS_STATS_TRAIN_FILE


# ----------------------------------------------------------------------------------------------------------------------
def identity_collate(
        batch: Any
) -> Any:
    """Identity collate function"""

    return batch


# ======================================================================================================================
class ShotStatsDataset(Dataset):
    """
    Dataset to get shot stats.

    Attributes
    ----------
    dataset : MastDataset
        Target dataset.
    sources_signals : list
        List of source-signal items.

    Methods
    -------
    __len__()
        Return the size of the dataset.
    __getitem__(idx)
        Return shot stats by shot index.

    """

    # ------------------------------------------------------------------------------------------------------------------
    def __init__(
            self,
            dataset: MastDataset
    ) -> None:
        """
        Initialize class attributes

        Parameters
        ----------
        dataset : MastDataset
            Input dataset.

        Returns
        -------
        None

        """

        self.dataset = dataset
        self.sources_signals = list(dataset[0].keys())

    # ------------------------------------------------------------------------------------------------------------------
    def __len__(
            self
    ) -> int:
        """
        Return the size of the dataset.

        Returns
        -------
        int
            Length of self.dataset.

        """

        return len(self.dataset)

    # ------------------------------------------------------------------------------------------------------------------
    def __getitem__(
            self,
            idx: int
    ) -> dict[str, Any]:
        """
        Return shot stats by shot index.

        Parameters
        ----------
        idx : int
            Shot index.

        Returns
        -------
        dict[str, Any]
            Dictionary with shot statistics.

        """

        shot = self.dataset[idx]
        shot_id = self.dataset.get_shot_id(idx=idx)
        stats: dict[str, Any] = {"shot_idx": idx, "shot_id": shot_id}

        for var in self.sources_signals:

            values = shot[var].get("values", [])

            # Initialize nested dict
            stats[var] = {
                "n_dim_shot": None,
                "mean": None,
                "variance": None
            }

            if len(values) > 0:

                mean_sample = np.nanmean(values)
                # std_sample = np.nanstd(values)  # Unbiased std NOT REALLY  # FIXME: Is this needed? [Cecle]
                variance_sample = np.nanvar(values, ddof=0)  # Unbiased variance

                # Handle all-NaN arrays
                if np.isnan(mean_sample):
                    mean_sample = None
                if np.isnan(variance_sample):
                    variance_sample = None

                stats[var]["n_dim_shot"] = values.size if hasattr(values, "size") else None
                stats[var]["mean"] = float(mean_sample) if mean_sample is not None else None
                stats[var]["variance"] = float(variance_sample) if variance_sample is not None else None

        return stats


# ----------------------------------------------------------------------------------------------------------------------
def compute_mean_std_per_shot_to_csv(
        dataset: MastDataset,
        csv_path: str,
        num_workers: int = 32,
        batch_size: int = 32
) -> None:
    """
    Compute Mean and STD per shot in provided dataset, and save results to target CSV file.

    Parameters
    ----------
    dataset : MastDataset
        Target MAST dataset.
    csv_path : str
        Target CSV path.
    num_workers : int
        Target number of workers.
        Optional. Default: 32.
    batch_size : int
        Target batch size.
        Optional. Default: 32.

    Returns
    -------
    None

    """

    stats_dataset = ShotStatsDataset(dataset=dataset)

    loader = DataLoader(
        dataset=stats_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=identity_collate,
        persistent_workers=num_workers > 0,
    )

    # Ensure directory exists
    os.makedirs(
        name=os.path.dirname(p=csv_path),
        exist_ok=True
    )

    # Overwrite CSV
    with open(csv_path, "w", newline="") as ff:
        writer = csv.writer(ff)

        # Write header
        header = ["shot_idx", "shot_id", "variable", "n_dim_shot", "mean", "variance"]
        writer.writerow(header)

        counter = 0
        for batch in loader:
            for shot_stats in batch:

                counter += 1
                if counter % 10 == 0:
                    print(f"Processed {counter} shots")

                for var in stats_dataset.sources_signals:
                    row = [
                        shot_stats["shot_idx"],
                        shot_stats["shot_id"],
                        var,
                        shot_stats[var].get("n_dim_shot", None),    # Safe fallback
                        shot_stats[var].get("mean", None),          # Safe fallback
                        shot_stats[var].get("variance", None)       # Safe fallback
                    ]
                    writer.writerow(row)

            # Optional: flush every batch
            ff.flush()
            os.fsync(ff.fileno())


# ======================================================================================================================
if __name__ == "__main__":

    print(f"Number of available CPU cores: {cpu_count()}\n")
    mp.set_start_method(method="spawn", force=True)

    # ------------------------------------------------------------------------------------------------------------------
    # Argument parsing
    # ------------------------------------------------------------------------------------------------------------------

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default=default_config,
        help="Path to the config_get_metadata YAML file.",
    )
    parser.add_argument(
        "--shots_stats_val_path",
        type=str,
        default=default_shots_stats_val_file,
        help="Path to the CSV file where shots statistics for validation will be saved."
    )
    parser.add_argument(
        "--shots_stats_test_path",
        type=str,
        default=default_shots_stats_test_file,
        help="Path to the CSV file where shots statistics for testing will be saved."
    )
    parser.add_argument(
        "--shots_stats_train_path",
        type=str,
        default=default_shots_stats_train_file,
        help="Path to the CSV file where shots statistics for training will be saved."
    )
    parser.add_argument(
        "--use_std_scaling",
        action="store_true",
        help="Activate STD scaling. If not provided, it defaults to `use_std_scaling = False`."
    )
    parser.add_argument(
        "--skip_incomplete_shots",
        action="store_true",
        help="Skip incomplete shots. If not provided, it defaults to `skip_incomplete_shots = False`, which in turn "
             "results in `return_incomplete_shots = True`."
    )
    parser.add_argument(
        "--remove_outliers",
        action="store_true",
        help="Remove outliers. If not provided, it defaults to `remove_outliers = False`."
    )
    parser.add_argument(
        "--store_manager_settings",
        type=json.loads,
        help="User-defined store manager settings for the target MAST_dataset instance as defined in "
             "`src.MAST_tools.MAST_dataset.StoreManagerParameters`, passed as (keyword, value) pairs in a JSON object. "
             "It is useful to provide store manager settings different than the default ones defined in "
             "`src.MAST_tools.store_utils.MASTStorageManager.__init__`.",
        default='{\"base_local_zarr_path\":\"' + DEFAULT_BASE_LOCAL_ZARR_PATH + '\"}'
    )

    args, _ = parser.parse_known_args()

    # Trick to default boolean parameters to True
    args.return_incomplete_shots = not args.skip_incomplete_shots

    # Load Task YAML config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # ------------------------------------------------------------------------------------------------------------------
    # Specific Data Preprocessing for LCFS profiles
    # ------------------------------------------------------------------------------------------------------------------

    train_shots_, test_shots_, val_shots_ = get_train_test_val_shots(
        max_index=config["max_shot_index"],
        shuffle=config["shuffle"],
        seed=config["seed"]
    )

    local_flag = config["local"]

    # ------------------------------------------------------------------------------------------------------------------
    # Create unstandardized val dataset
    # ------------------------------------------------------------------------------------------------------------------

    print("\nProcessing val dataset...")

    preprocessing_val_dataset = initialize_MAST_dataset( 
        config_task=config,
        shots_list=val_shots_,
        local_flag=local_flag,
        use_std_scaling=args.use_std_scaling,                   # It defaults to False
        return_incomplete_shots=args.return_incomplete_shots,   # t defaults to True
        remove_outliers=args.remove_outliers,                   # It defaults to False
        store_manager_settings=args.store_manager_settings
    )

    compute_mean_std_per_shot_to_csv(
        dataset=preprocessing_val_dataset,
        csv_path=args.shots_stats_val_path,
        num_workers=20,     # Default value "32" may trigger warnings depending on local resources.
        batch_size=20       # Default value "32" may trigger warnings depending on local resources.
    )

    print("... Done.")

    # ------------------------------------------------------------------------------------------------------------------
    # Create unstandardized test dataset
    # ------------------------------------------------------------------------------------------------------------------

    print("\nProcessing test dataset...")

    preprocessing_test_dataset = initialize_MAST_dataset( 
        config_task=config,
        shots_list=test_shots_,
        local_flag=local_flag,
        use_std_scaling=args.use_std_scaling,                   # It defaults to False
        return_incomplete_shots=args.return_incomplete_shots,   # It defaults to True
        remove_outliers=args.remove_outliers,                   # It defaults to False
        store_manager_settings=args.store_manager_settings
    )

    compute_mean_std_per_shot_to_csv(
        dataset=preprocessing_test_dataset,
        csv_path=args.shots_stats_test_path,
        num_workers=20,     # Default value "32" may trigger warnings depending on local resources.
        batch_size=20       # Default value "32" may trigger warnings depending on local resources.
    )

    print("... Done.")

    # ------------------------------------------------------------------------------------------------------------------
    # Create unstandardized train dataset
    # ------------------------------------------------------------------------------------------------------------------

    print("\nProcessing train dataset...")

    preprocessing_train_dataset = initialize_MAST_dataset( 
        config_task=config,
        shots_list=train_shots_,
        local_flag=local_flag,
        use_std_scaling=args.use_std_scaling,                   # It defaults to False
        return_incomplete_shots=args.return_incomplete_shots,   # It defaults to True
        remove_outliers=args.remove_outliers,                   # It defaults to False
        store_manager_settings=args.store_manager_settings
    )

    compute_mean_std_per_shot_to_csv(
        dataset=preprocessing_train_dataset,
        csv_path=args.shots_stats_train_path,
        num_workers=20,     # Default value "32" may trigger warnings depending on local resources.
        batch_size=20       # Default value "32" may trigger warnings depending on local resources.
    )

    print("... Done.")

    print("\nAll computations done.")
