from MAST_signal import SIGNAL
import MAST_store as MASTbucket
import xarray as xr
import random
from sklearn.impute import SimpleImputer
import os
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import json


CONFIG_FILE = "config_pca.json"
def load_config(path):
    with open(path, "r") as f:
        config = json.load(f)
    return config

def shuffle_shot_ids(shot_ids, N):
    random.seed(42)
    random.shuffle(shot_ids)
    return shot_ids[:N]


def imputed_array(mast_signal, store):
    """
    Process a single mast_signal to handle missing data, 
    and return the imputed array.

    Args:
        mast_signal: SIGNAL from MAST_signal.py.

    Returns:
        Imputed numpy array of flux loop values for the given shot ID.
    """
    vals = mast_signal.get_values(store)
    
    imp = SimpleImputer(missing_values=np.nan, strategy="mean")
    imp.fit(vals)
    vals = imp.transform(vals)
    
    return vals

def process_single_shot_id(shot_id, group, signal_name):
    sig = SIGNAL(group, signal_name, shot_id)
    store = MASTbucket.make_store(shot_id)
    return imputed_array(sig, store)

def main():
    # input
    config = load_config(CONFIG_FILE)

    # Extract parameters
    n_components = config.get("n_components")
    group = config.get("group")
    signal_name = config.get("signal_name")
    N = config.get("nr_shots")

    # Get all shot ids
    shot_ids = MASTbucket.list_all_shots()

    # Return N rnd shot_ids
    random_shot_ids = shuffle_shot_ids(shot_ids,N)

    # Process shot ids
    results = []
    for shot_id in random_shot_ids:
        vals = process_single_shot_id(shot_id, group, signal_name)
        if vals is not None:
            results.append(vals)

    # Concatenate the results
    if results:
        flux_loop_array = np.concatenate([*results], axis=1)
    else:
        print("Warning: No results to concatenate.")

    flux_loop_array = flux_loop_array.T


    pca = PCA(n_components)
    pca.fit(flux_loop_array)

    explained_variance = pca.explained_variance_ratio_
    cumulative_explained_variance = 0

    for i, val in enumerate(explained_variance):
        cumulative_explained_variance += val
        print(f"Principal component: {i+1}")
        print(f"Explained variance: {val:.3f}")
        print(f"Cumulative explained variance: {cumulative_explained_variance:.3f}\n") 

    # Demonstrate PCA transform and reconstruction 
    id =16092
    vals = process_single_shot_id(id, group, signal_name)

    transformed = pca.transform(vals.T)
    reconstructed = pca.inverse_transform(transformed).T

    fig, axes = plt.subplots(nrows=2, figsize=(8, 10))  # Adjust figsize as needed

    # First subplot: Original
    axes[0].set_title("Original")
    p0 = axes[0].pcolorfast(vals)
    fig.colorbar(p0, ax=axes[0])

    # Second subplot: Reconstructed
    axes[1].set_title("Reconstructed")
    p1 = axes[1].pcolorfast(reconstructed)
    fig.colorbar(p1, ax=axes[1])

    # Layout and save
    plt.tight_layout()
    fig.savefig("comparison.png", bbox_inches='tight', dpi=300)
    plt.show()


if __name__ == "__main__":
    main()

