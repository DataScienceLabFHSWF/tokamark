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


def collate_preprocessing (batch):
    
    sources_signals = batch[0].keys()

    formatted_batch = {var: [] for var in sources_signals}
    
    for data in batch :
        for var, data_var in data.items():
            if len(data_var['values']) != 0 :
                formatted_batch[var].append(data_var['values'])

    formatted_batch_mean_std = {var: [] for var in sources_signals}

    for var in sources_signals:
        shapes = np.array([ar.shape for ar in [arr[...,-1] for arr in formatted_batch[var]]])
        uniq = np.unique(shapes, axis=0)
        # print(shapes)
        if len(uniq)!=1:
            formatted_batch_mean_std[var] = [0, 0, 0]
        else:
            formatted_batch_mean_std[var] = [ len(formatted_batch[var]), 
                                            np.nanmean( np.stack( [np.nanmean(arr, axis=-1) for arr in formatted_batch[var]] ), axis = 0),
                                            np.nanmean( np.stack( [np.nanstd(arr, axis=-1) for arr in formatted_batch[var]] ), axis = 0) ]
    
    return(formatted_batch_mean_std)
    

def compute_mean_std(dataset, 
                     batch_size, 
                     num_workers):

    sources_signals = dataset[0].keys()

    n_samples = { var : 0 for var in sources_signals }
    sum_mean = { var : 0 for var in sources_signals }
    sum_std = { var : 0 for var in sources_signals }

    loader = DataLoader(dataset, 
                        batch_size=batch_size,
                        num_workers=num_workers,
                        shuffle=False,
                        collate_fn=collate_preprocessing)

    for batch in loader:

        for var in sources_signals:

            n_batch = batch[var][0]
            mean_batch = batch[var][1]
            std_batch = batch[var][2]

            sum_mean[var] += mean_batch * n_batch
            sum_std[var] += std_batch * n_batch
            n_samples[var] += n_batch

    mean = { var : 0.0 for var in sources_signals }
    std = { var : 0.0 for var in sources_signals }

    for var in sources_signals:
        mean[var] = sum_mean[var] / n_samples[var]
        std[var] = sum_std[var] / n_samples[var]

    flattened_dict_mean = mean
    flattened_dict_std = std

    for var in sources_signals:
        # print( flattened_dict_mean[var].shape )
        mean_value = np.nanmean( mean[var] ) 
        mean_arr = np.full_like(mean[var], mean_value)
        flattened_dict_mean[var] = mean_arr
        # print( flattened_dict_std[var].shape )
        std_value = np.nanmean( std[var] ) 
        std_arr = np.full_like(std[var], std_value)
        flattened_dict_std[var] = std_arr
        # find nans
        where_are_NaNs = np.isnan(flattened_dict_mean[var])
        flattened_dict_mean[var][where_are_NaNs] = 0
        flattened_dict_std[var][where_are_NaNs] = 1

    return mean, std


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
        default="/scripts/preprocessing/config_get_metadata.yaml",
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

    # ..................................................................................................................
    # Create unstandardized train dataset 
    preprocessing_train_dataset = initialize_MAST_dataset( 
        config,
        train_shots_,
        use_std_scaling = False,
        return_incomplete_shots=True
    )

    # ..................................................................................................................
    # Create unstandardized train dataset 

    dict_mean, dict_std = compute_mean_std( preprocessing_train_dataset, 
                                 batch_size=config['dataloader_setting']['batch_size'], 
                                 num_workers=config['dataloader_setting']['num_workers'])

    
    for i, sample in enumerate(preprocessing_train_dataset):
        
        max_samples=100
        if i >= max_samples:
            raise ValueError("❌ No valid sample found within limit.")

        # Check that each signal has a non-empty time array
        valid = all(len(signal.get("time", [])) > 1 for signal in sample.values())
        if not valid:
            continue  # Skip invalid sample

        # Found a valid sample
        dict_metadata = {}
        for var, signal in sample.items():
            time = np.array(signal["time"])
            values = np.array(signal["values"])

            # Compute median dt
            dt = round(np.median(np.diff(time)), 6) if len(time) > 1 else None

            dict_metadata[var] = {
                "dt": dt,
                "values_shape": values.shape[:-1],  # exclude time dimension
                "mean": dict_mean[var],
                "std": dict_std[var],
            }
            
    out_dir = Path("artifacts")
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / 'dict_metadata.pkl', 'wb') as f_:
            pickle.dump(dict_metadata, f_)


    
