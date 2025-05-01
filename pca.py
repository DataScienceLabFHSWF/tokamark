import joblib
import json
import matplotlib.pyplot as plt
from MAST_signal import SIGNAL
import MAST_store as MASTbucket
from multiprocessing import Pool
import numpy as np
import random
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from tqdm import tqdm
import zarr


def load_config(path):
    with open(path, "r") as f:
        config = json.load(f)
    return config

def shuffle_shot_ids(shot_ids, N, seed=None):
    random.seed(seed)
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


def process_single_shot_id(shot_id, group, signal_name, location):
    sig = SIGNAL(group, signal_name, shot_id)
    store = MASTbucket.make_store(shot_id, location=location)
    return imputed_array(sig, store)


def process_single_shot_id_star(args):
    """Wrapper for process_single_shot_id for multiprocessing starmap."""
    return process_single_shot_id(*args)


def main():
    CONFIG_FILE = "config_pca.json"
    
    # input
    config = load_config(CONFIG_FILE)

    # Extract parameters
    max_components = config.get("max_components")
    group = config.get("group")
    signal_name = config.get("signal_name")
    N = config.get("nr_shots")
    processes = config.get("processes")
    location = config.get("location")

    # Get all shot ids
    shot_ids = MASTbucket.list_all_shots(location=location)

    # Return N random shot_ids
    random_shot_ids = shuffle_shot_ids(shot_ids, N)

    # Process shot ids
    with Pool(processes=processes) as pool:
        args_iterable = [(shot_id, group, signal_name, location) for shot_id in random_shot_ids]
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
    }, "pca_model.joblib")

def test_reconstruction(id = 16092):

    CONFIG_FILE = "config_pca.json"
    config = load_config(CONFIG_FILE)
    location = config.get("location")

    # Demonstrate PCA transform and reconstruction
    models = joblib.load("pca_model.joblib")
    pca = models["pca"]
    scaler = models["scaler"]

    group = models["group"]
    signal_name = models["signal_name"]

    vals = process_single_shot_id(id, group, signal_name, location)
    vals_scaled = scaler.transform(vals.T)

    transformed = pca.transform(vals_scaled)

    reconstructed = pca.inverse_transform(transformed)
    reconstructed = scaler.inverse_transform(reconstructed)


    fig, axes = plt.subplots(nrows=2, figsize=(8, 10))

    axes[0].set_title("Original")
    p0 = axes[0].pcolorfast(vals)
    fig.colorbar(p0, ax=axes[0])

    axes[1].set_title("Reconstructed")
    p1 = axes[1].pcolorfast(reconstructed.T)
    fig.colorbar(p1, ax=axes[1])

    plt.tight_layout()
    fig.savefig("comparison.png", bbox_inches="tight", dpi=300)
    plt.show()

    RMS = np.sqrt(np.mean((vals - reconstructed.T) ** 2))
    print(f"RMS = {RMS}")

if __name__ == "__main__":
    main()
    test_reconstruction()
