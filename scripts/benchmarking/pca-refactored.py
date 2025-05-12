# Refactored by: Rodrigo Ordonez-Hurtado (rodrigo.ordonez.hurtado@ibm.com)

import random
from sklearn.impute import SimpleImputer
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import json
from tqdm import tqdm
from multiprocessing import Pool

import sys
sys.path.insert(1, '../MAST_tools')

from signal_utils import MASTSignalManager  # noqa
from store_utils import MASTStorageManager  # noqa


# ----------------------------------------------------------------------------------------------------------------------
def load_config(path):
    with open(path, "r") as f:
        config = json.load(f)
    return config


# ----------------------------------------------------------------------------------------------------------------------
def shuffle_shot_ids(shot_ids, n, seed=None):
    random.seed(seed)
    random.shuffle(shot_ids)
    return shot_ids[:n]


# ----------------------------------------------------------------------------------------------------------------------
def impute_array(signal_manager, store):
    """
    Process a single mast_signal to handle missing data, and return the imputed array.

    Args:
        signal_manager: Instance of MASTSignalManager class.

        store: Target store.

    Returns:
        Imputed numpy array of flux loop values for the given shot ID.
    """
    vals = signal_manager.get_values(store)

    imp = SimpleImputer(missing_values=np.nan, strategy="mean")
    imp.fit(vals)
    vals = imp.transform(vals)

    return vals


# ----------------------------------------------------------------------------------------------------------------------
def process_single_shot_id(shot_id, source, signal_name, local=False):
    signal_manager = MASTSignalManager(source=source, signal_name=signal_name, shot_id=shot_id)

    store_manager = MASTStorageManager()
    store = store_manager.make_shot_store(shot_id=30471, local=local)

    return impute_array(signal_manager, store)


# ----------------------------------------------------------------------------------------------------------------------
def process_single_shot_id_star(args):
    """Wrapper for process_single_shot_id for multiprocessing starmap."""
    return process_single_shot_id(*args)


# ----------------------------------------------------------------------------------------------------------------------
def main():

    # Input
    config = load_config("config_pca.json")

    # Extract parameters
    n_components = config.get("n_components")
    group = config.get("group")
    signal_name = config.get("signal_name")
    n_shots = config.get("nr_shots")
    processes = config.get("processes")
    local = False

    # Get all shot ids
    store_manager = MASTStorageManager()
    shot_ids = store_manager.list_all_shots(local=local)

    # Return N random shot_ids
    random_shot_ids = shuffle_shot_ids(shot_ids, n_shots)

    # Process shot ids
    with Pool(processes=processes) as pool:
        args_iterable = [(shot_id, group, signal_name) for shot_id in random_shot_ids]
        results = list(tqdm(
            pool.imap(process_single_shot_id_star, args_iterable),
            total=len(random_shot_ids)
        ))

    # Concatenate the results
    data_array = np.ndarray([])
    if results:
        data_array = np.concatenate([*results], axis=1)
    else:
        print("Warning: No results to concatenate.")
    data_array = data_array.T

    pca = PCA(n_components)
    pca.fit(data_array)

    explained_variance = pca.explained_variance_ratio_
    cumulative_explained_variance = 0

    for i, val in enumerate(explained_variance):
        cumulative_explained_variance += val
        print(f"Principal component: {i+1}")
        print(f"Explained variance: {val:.3f}")
        print(f"Cumulative explained variance: {cumulative_explained_variance:.3f}\n")

    # Demonstrate PCA transform and reconstruction
    shot_id = 16092
    vals = process_single_shot_id(shot_id, group, signal_name)

    transformed = pca.transform(vals.T)
    reconstructed = pca.inverse_transform(transformed).T

    fig, axes = plt.subplots(nrows=2, figsize=(8, 10))

    axes[0].set_title("Original")
    p0 = axes[0].pcolorfast(vals)
    fig.colorbar(p0, ax=axes[0])

    axes[1].set_title("Reconstructed")
    p1 = axes[1].pcolorfast(reconstructed)
    fig.colorbar(p1, ax=axes[1])

    plt.tight_layout()
    fig.savefig("comparison.png", bbox_inches="tight", dpi=300)
    plt.show()


# ======================================================================================================================
if __name__ == "__main__":
    main()
