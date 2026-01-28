import csv
import os
import sys

import argparse
import yaml
import numpy as np
import torch.multiprocessing as mp

from MAST_benchmark.data import initialize_MAST_dataset
from MAST_benchmark.data_split import get_train_test_val_shots

import numpy as np
import torch
from torch.utils.data import Dataset
import os
import csv
from torch.utils.data import DataLoader

def identity_collate(batch):
    return batch

class ShotStatsDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.sources_signals = list(dataset[0].keys())

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        shot = self.dataset[idx]
        shot_id = self.dataset.get_shot_id(idx)

        stats = {
            "shot_idx": idx,
            "shot_id": shot_id,
        }

        for var in self.sources_signals:
            values = shot[var]["values"]

            # initialize nested dict
            stats[var] = {
                "n_dim_shot":None,
                "mean": None,
                "variance": None,
            }

            if len(values) > 0:
                mean_sample = np.nanmean(values)
                # std_sample = np.nanstd(values) # unbiaised std NOT REALLY
                variance_sample = np.nanvar(values, ddof=0) # unbiaised variance

                # handle all-NaN arrays
                if np.isnan(mean_sample):
                    mean_sample = None
                if np.isnan(variance_sample):
                    variance_sample = None


                stats[var]["n_dim_shot"] = values.size if hasattr(values, "size") else None
                stats[var]["mean"] = float(mean_sample) if mean_sample is not None else None
                stats[var]["variance"] = float(variance_sample) if variance_sample is not None else None

        return stats

import os
import csv
from torch.utils.data import DataLoader

def compute_mean_std_per_shot_to_csv(dataset, csv_path, 
                                    #  num_workers=8,
                                    #  batch_size=32
                                     num_workers=32,
                                     batch_size=32
                                     ):
    
    stats_dataset = ShotStatsDataset(dataset)

    loader = DataLoader(
        stats_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=identity_collate,
        persistent_workers=num_workers > 0,
    )

    # ensure directory exists
    # csv_dir = os.path.dirname(csv_path)
    # os.makedirs(csv_dir, exist_ok=True)

    # overwrite CSV
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)

        # write header
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
                        shot_stats[var].get("n_dim_shot", None),  # safe fallback
                        shot_stats[var]["mean"],
                        shot_stats[var]["variance"]
                    ]
                    writer.writerow(row)

            # optional: flush every batch
            f.flush()
            os.fsync(f.fileno())




# ======================================================================================================================
if __name__ == "__main__":

    # print(f"Number of available CPU cores: {cpu_count()}\n")
    mp.set_start_method("spawn", force=True)

    # -------------------------------------------------------------------
    # Argument parsing
    # -------------------------------------------------------------------
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="/home/ir-rous1/rds/rds-ukaea-ap002-mOlK9qn0PlQ/ir-rous1/output/cnn-baseline/fairmast-data-preprocessing/scripts/preprocessing/config_get_metadata.yaml",
        # default="/config_get_metadata.yaml",
        help="Path to the task YAML config file",
    )
    args, _ = parser.parse_known_args()

    # Load Task YAML config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # ..................................................................................................................
    # Specific Data Preprocessing for LCFS profiles

    train_shots_, test_shots_, val_shots_ = get_train_test_val_shots(
        max_index=config["subset_of_shots"]
    )

    local_flag= config["local"]

    # ..................................................................................................................
    # Create unstandardized val dataset 
    preprocessing_val_dataset = initialize_MAST_dataset( 
        config,
        val_shots_,
        local_flag,
        use_std_scaling = False,
        return_incomplete_shots=True
    )

    csv_path = "shot_statistics_val_NEW.csv"
    compute_mean_std_per_shot_to_csv(
        preprocessing_val_dataset,
        csv_path=csv_path, 
    )

    # ..................................................................................................................
    # Create unstandardized test dataset 
    preprocessing_test_dataset = initialize_MAST_dataset( 
        config,
        test_shots_,
        local_flag,
        use_std_scaling = False,
        return_incomplete_shots=True
    )

    csv_path = "shot_statistics_test_NEW.csv"
    compute_mean_std_per_shot_to_csv(
        preprocessing_test_dataset,
        csv_path=csv_path,
    )

    # ..................................................................................................................
    # Create unstandardized train dataset 
    preprocessing_train_dataset = initialize_MAST_dataset( 
        config,
        train_shots_,
        local_flag,
        use_std_scaling = False,
        return_incomplete_shots=True
    )

    csv_path = "shot_statistics_train_NEW.csv"
    # compute_mean_std_per_shot_to_csv(preprocessing_train_dataset, csv_path)
    
    compute_mean_std_per_shot_to_csv(
        preprocessing_train_dataset,
        csv_path=csv_path,
    )
