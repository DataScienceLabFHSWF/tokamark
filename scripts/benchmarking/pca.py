import joblib
import json
import matplotlib.pyplot as plt
from multiprocessing import Pool
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import zarr
import os

import sys
cwd = os.path.dirname(os.path.abspath(__file__))
mother_dir = os.path.dirname(cwd) + os.sep
sys.path.append(os.path.abspath(os.path.join(mother_dir , "MAST_tools")))
sys.path.append(mother_dir)

from sigfill import SimpleFiller
from signal_utils import MASTSignalManager  
from store_utils import MASTStorageManager
from  utils import (shuffle_shot_ids)

def load_config(path):
    with open(path, "r") as f:
        config = json.load(f)
    return config

def process_single_shot_id( 
    shot_id,
    group,
    signal_name, 
    store_manager,
    local
    ):
    sig = MASTSignalManager(group, signal_name, shot_id)
    store= store_manager.make_shot_store(shot_id=shot_id, local=local)

    imputer = SimpleFiller(sig, store)
    return imputer.simple_fill()

def process_single_shot_id_conditional(local,
                            group,
                            signal_name,
                            shot_ids,
                            focused_shot_index):

    from .sigfill import  test_conditional_fill
    channel_names, conditional_filled = test_conditional_fill(
                                        local, 
                                        group, 
                                        signal_name, 
                                        shot_ids,
                                        focused_shot_index
                                        )
                                        
    conditional_filled = conditional_filled[0].T
    conditional_filled = conditional_filled.to_numpy()
    return conditional_filled


def process_single_shot_id_star(args):
    """Wrapper for process_single_shot_id for multiprocessing starmap."""
    return process_single_shot_id(*args)


def main(
    config_file, 
    local
    ):

    # input
    config = load_config(config_file)

    # Extract parameters from config file
    max_components = config.get("max_components")
    group = config.get("group")
    signal_name = config.get("signal_name")
    N = config.get("nr_shots")
    processes = config.get("processes")

    # Instanciate MASTStorageManager
    store_manager = MASTStorageManager()

    # Get all shot ids
    shot_ids = store_manager.list_all_shots(local=local)

    # Return N random shot_ids
    random_shot_ids = shuffle_shot_ids(shot_ids)[:N]
    
    if 16092 in random_shot_ids:
        print(f"YES {len(random_shot_ids)}")
    else:
        print(f"NO {len(random_shot_ids)}")

    # Process shot ids
    with Pool(processes=processes) as pool:
        args_iterable = [(  shot_id, 
                            group, 
                            signal_name, 
                            store_manager,
                            local) for shot_id in random_shot_ids]
        results = list(tqdm(
            pool.imap(process_single_shot_id_star, args_iterable),
            total=len(random_shot_ids)
        ))

    if results:
        data_array = np.concatenate([*results], axis=1).T
    else:
        print("Warning: No results to concatenate.")

    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data_array)

    for n_components in range(1,max_components):
        pca = PCA(n_components)
        pca.fit(scaled_data)

        explained_variance = pca.explained_variance_ratio_

        if np.cumsum(explained_variance)[-1] >= 0.997:
            break
    
    # Save PCA and scaler
    joblib.dump({
        "pca": pca,
        "scaler": scaler,
        "group": group,
        "signal_name": signal_name
    },  "pca_model.joblib")

def test_reconstruction(shot_id, config_file, local):

    # Instanciate MASTStorageManager
    store_manager = MASTStorageManager()

    config = load_config(config_file)
    location = config.get("location")

    # Demonstrate PCA transform and reconstruction
    models = joblib.load("pca_model.joblib")
    pca = models["pca"]
    scaler = models["scaler"]

    group = models["group"]
    signal_name = models["signal_name"]

    vals = process_single_shot_id(
        shot_id,
        group,
        signal_name, 
        store_manager,
        local
        )
    vals_scaled = scaler.transform(vals.T)

    transformed = pca.transform(vals_scaled)

    reconstructed = pca.inverse_transform(transformed)
    reconstructed = scaler.inverse_transform(reconstructed)

    RMS = np.sqrt(np.mean((vals - reconstructed.T) ** 2))
    print(f"RMS = {RMS}")

    fig, axes = plt.subplots(nrows=2, figsize=(8, 10))

    vmin = min(vals.min(), reconstructed.T.min())
    vmax = max(vals.max(), reconstructed.T.max())

    p0 = axes[0].pcolorfast(vals, vmin=vmin, vmax=vmax)
    p1 = axes[1].pcolorfast(reconstructed.T, vmin=vmin, vmax=vmax)


    axes[0].set_title(r"$\bf{Original}$" + f" {group}/{signal_name} shot id = {shot_id}")
    fig.colorbar(p0, ax=axes[0])

    axes[1].set_title(r"$\bf{Reconstructed}$" + f" (RMS = {RMS:.4f})")
    fig.colorbar(p1, ax=axes[1])

    plt.tight_layout()


    fig.savefig(f"{group}_{signal_name}.png", bbox_inches="tight", dpi=300)
    plt.show()

    old_filename = "pca_model.joblib"
    new_filename = f"{group}_{signal_name}_pca_model.joblib"
    os.rename(old_filename, new_filename)

if __name__ == "__main__":
    main("scripts/benchmarking/config_pca.json",True)
    test_reconstruction(16092, "scripts/benchmarking/config_pca.json",True)
