import argparse
import yaml
import pickle
import numpy as np
import torch.multiprocessing as mp

from multiprocessing import cpu_count
from pathlib import Path
from torch.utils.data import DataLoader

from MAST_benchmark.data import initialize_MAST_dataset
from MAST_benchmark.data_split import get_train_test_val_shots

def to_python(obj):
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, tuple):
        return list(obj)
    if isinstance(obj, dict):
        return {k: to_python(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_python(v) for v in obj]
    return obj

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
    # load mean_std_train.yaml
    with open("./artifacts/stats_mean_std/mean_std_train.yaml") as f:
        dict_mean_std = yaml.safe_load(f)
    # print(dict_mean_std)

    # ..................................................................................................................
    # Create unstandardized train dataset 
    preprocessing_train_dataset = initialize_MAST_dataset( 
        config,
        train_shots_,
        local_flag,
        use_std_scaling = False,
        return_incomplete_shots=True
    )

    # ..................................................................................................................
    # Create unstandardized train dataset 

    dict_metadata = {}
    max_samples = 100

    # Get expected variable count
    first_sample = next(iter(preprocessing_train_dataset))
    target_vars = set(first_sample.keys())

    for i, sample in enumerate(preprocessing_train_dataset):
        # print(i)

        if i >= max_samples:
            break  # stop after limit, but keep what we collected

        for var, signal in sample.items():

            # Only compute once per variable
            if var in dict_metadata:
                continue

            time = np.array(signal.get("time", []))
            values = np.array(signal.get("values", []))

            # Skip invalid signal
            if len(time) == 0 or len(values) == 0:
                print("Skipping var", var)
                continue
            
            # print("\nSaving var", var)

            # Compute median dt
            dt = round(np.median(np.diff(time)), 6)

            dict_metadata[var] = {
                "dt": dt,
                "values_shape": values.shape[:-1],  # exclude time dimension
                "mean": dict_mean_std[var]["mean"]["no_outliers_z6"], 
                "std": dict_mean_std[var]["std"]["no_outliers_z6"],
            }
        
        # ✅ Stop once all variables are filled
        if set(dict_metadata.keys()) == target_vars:
            print("✅ dict_metadata fully filled. Stopping.")
            break

    # Optional safety check
    if len(dict_metadata) == 0:
        raise ValueError("❌ No valid signals found within limit.")

    clean_dict = to_python(dict_metadata)

    out_dir = Path("artifacts")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    with open(out_dir / "dict_stats_metadata.yaml", "w") as f_:
        yaml.dump(clean_dict, f_, sort_keys=False)



    
