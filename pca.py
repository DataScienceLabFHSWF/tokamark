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


def process_single_shot_id(shot_id, group, signal_name):
    sig = SIGNAL(group, signal_name, shot_id)
    store = MASTbucket.make_store(shot_id)
    return imputed_array(sig, store)


def process_single_shot_id_star(args):
    """Wrapper for process_single_shot_id for multiprocessing starmap."""
    return process_single_shot_id(*args)


def main():
    CONFIG_FILE = "config_pca.json"
    
    # input
    config = load_config(CONFIG_FILE)

    # Extract parameters
    max_componentss = config.get("max_components")
    group = config.get("group")
    signal_name = config.get("signal_name")
    N = config.get("nr_shots")
    processes = config.get("processes")

    # Get all shot ids
    shot_ids = MASTbucket.list_all_shots()

    # Return N random shot_ids
    random_shot_ids = shuffle_shot_ids(shot_ids, N)

    # Process shot ids
    with Pool(processes=processes) as pool:
        args_iterable = [(shot_id, group, signal_name) for shot_id in random_shot_ids[:10]]
        results = list(tqdm(
            pool.imap(process_single_shot_id_star, args_iterable),
            total=len(random_shot_ids)
        ))

    # Concatenate the results
    if results:
        data_array = np.concatenate([*results], axis=1)
    else:
        print("Warning: No results to concatenate.")

    data_array = data_array.T

    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data_array)

    for n_components in range(1,max_componentss):
        pca = PCA(n_components)
        pca.fit(scaled_data)

        explained_variance = pca.explained_variance_ratio_
        cumulative_explained_variance = 0

        if np.cumsum(explained_variance) >= 99.7:
            break
    

    # Demonstrate PCA transform and reconstruction
    id = 16092
    vals = process_single_shot_id(id, group, signal_name)

    transformed = pca.transform(vals.T)
    reconstructed = pca.inverse_transform(transformed).T

    pca_results = {
        "original_data": data_array,
        "scaled_data": scaled_data,
        "scaler": scaler,
        "loading": pca.components_,
        "explained_variance_ratio": explained_variance,
        "principal_component_scores": principal_component_scores,
        "n_components": pca.n_components_,
        "transformed": transformed
    }

    joblib.dump(pca_results, "PCA_joblib")
    print(f"PCA results saved to PCA_joblib")


    # 4. Save the array-like results in Zarr format
    zarr_dir= "pca_results.zarr"
    zarr.save(f"{zarr_dir}original_data", data_array)
    zarr.save(f"{zarr_dir}/scaled_data.zarr", scaled_data)
    zarr.save(f"{zarr_dir}/loadings.zarr", pca.components_)
    zarr.save(f"{zarr_dir}/explained_variance_ratio.zarr", explained_variance_ratio)
    zarr.save(f"{zarr_dir}/principal_component_scores.zarr", principal_component_scores)
    zarr.save(f"{zarr_dir}/transformed.zarr", transformed) 


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


if __name__ == "__main__":
    main()
