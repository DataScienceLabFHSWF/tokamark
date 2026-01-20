import csv
import os
import sys

import argparse
import yaml
import numpy as np
import torch.multiprocessing as mp

from MAST_benchmark.data import initialize_MAST_dataset
from MAST_benchmark.data_split import get_train_test_val_shots


def compute_mean_std_per_shot_to_csv(dataset, csv_path):

    sources_signals = dataset[0].keys()

    file_exists = os.path.isfile(csv_path)

    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)

        # write header only once
        if not file_exists:
            header = ["shot_idx", "shot_id"]
            for var in sources_signals:
                header += [f"{var}_mean", f"{var}_std"]
            writer.writerow(header)

        for shot_idx, shot in enumerate(dataset):
        
            shot_id = dataset.get_shot_id(shot_idx)
            row = [shot_idx, shot_id]

            for var in sources_signals:

                values = shot[var]["values"]

                if len(values) == 0:
                    row += [None, None]
                else:
                    mean_val = np.nanmean(values)
                    std_val = np.nanstd(values)

                    row += [mean_val, std_val]

            writer.writerow(row)
            f.flush()


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
        default="scripts/preprocessing/config_get_metadata.yaml",
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

    csv_path = "shot_statistics_val.csv"
    compute_mean_std_per_shot_to_csv(preprocessing_val_dataset, csv_path)

    # ..................................................................................................................
    # Create unstandardized test dataset 
    preprocessing_test_dataset = initialize_MAST_dataset( 
        config,
        test_shots_,
        local_flag,
        use_std_scaling = False,
        return_incomplete_shots=True
    )

    csv_path = "shot_statistics_test.csv"
    compute_mean_std_per_shot_to_csv(preprocessing_test_dataset, csv_path)

    # ..................................................................................................................
    # Create unstandardized train dataset 
    preprocessing_train_dataset = initialize_MAST_dataset( 
        config,
        train_shots_,
        local_flag,
        use_std_scaling = False,
        return_incomplete_shots=True
    )

    csv_path = "shot_statistics_train.csv"
    compute_mean_std_per_shot_to_csv(preprocessing_train_dataset, csv_path)