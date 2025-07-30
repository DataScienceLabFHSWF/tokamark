import joblib
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

cwd = os.path.dirname(os.path.abspath(__file__))
mother_dir = os.path.dirname(cwd) + os.sep
sys.path.append(os.path.abspath(os.path.join(mother_dir , "MAST_tools")))
sys.path.append(mother_dir)

from my_PCA import my_PCA, process_single_shot_id
from signal_utils import MASTSignalManager  
from store_utils import MASTStorageManager
from  utils import (shuffle_shot_ids)


my_pca = my_PCA("scripts/benchmarking/config_pca.json")

def fit(my_pca=my_pca, focused_shot_id = 16092, N=10, local = True):
    """ Prepare N random shots for PCA analysis """

    # Instanciate MASTStorageManager
    store_manager = MASTStorageManager()

    # Get all shot ids
    shot_ids = store_manager.list_all_shots(local=local)

    random_shot_ids = shuffle_shot_ids(shot_ids)[:N]
    
    # If PCA sample includes the shot we are going to use for testing, then remove it
    if focused_shot_id  in random_shot_ids:
        random_shot_ids.remove(focused_shot_id)
        print(f"YES {len(random_shot_ids)}")
    else:
        print(f"NO {len(random_shot_ids)}")

    # Return N random shot_ids
    my_pca.fit_PCA(local, random_shot_ids)

def test_reconstruction(shot_id, local):
    # Instanciate MASTStorageManager
    store_manager = MASTStorageManager()

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
    fit()
    test_reconstruction(16092, local=True)
