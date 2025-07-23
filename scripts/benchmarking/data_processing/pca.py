"""PCAnalysis class for performing PCA on MAST data.
"""

import joblib
import json
import matplotlib.pyplot as plt
from multiprocessing import Pool
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import os

import sys

cwd = os.path.dirname(os.path.abspath(__file__))
mother_dir = os.path.dirname(cwd)
scripts_dir = os.path.join(os.path.dirname(mother_dir))
sys.path.append(scripts_dir)

from MAST_tools.store_utils import MASTStorageManager
from sigfill import run_simple_filler
from utils import (read_data_split_csv, shuffle_shot_ids)



def process_single_shot_id( 
    shot_id:int,
    group:str,
    signal_name:str, 
    store_manager: MASTStorageManager,
    local: bool
    ):
    """Process a single shot_id to extract and fill the signal data.

    Parameters
    ----------
    shot_id : int
        shot id to process
    group : str
        name of the group to which the signal belongs
    signal_name : str
        name of the signal to process
    store_manager : MASTStorageManager
        instance of MASTStorageManager to handle data storage
    local : bool
        True if accessing data locally, False if accessing remotely

    Returns
    -------
    np.ndarray
        Processed data for the given shot_id
    """
    
    return run_simple_filler(
        shot_id,
        group,
        signal_name, 
        store_manager,
        local
        )
    
def process_single_shot_id_star(args):
    """Wrapper for process_single_shot_id for multiprocessing starmap."""
    return process_single_shot_id(*args)


class MASTpca:
    """PCAnalysis class for performing PCA on MAST data.
    """
    def __init__(self, config_file_path):
        self.config_file_path = config_file_path

    
    def _load_config(self):
        try:
            with open(self.config_file_path, "r") as f:
                # Load json
                config = json.load(f)

                # Extract parameters from config file
                config = {
                    "max_components" : config.get("max_components"),
                    "group" : config.get("group"),
                    "signal_name" : config.get("signal_name"),
                    "processes" : config.get("processes"),
                    "nr_shots" : config.get("nr_shots")
                    }

                return config

        except Exception as e:
            print(f"Error {e}")
            return None
    

    def fit_PCA(
        self,
        local: bool,
        shot_ids: list[int],
        output_directory
        ):
        """Aply PCA 

        Parameters
        ----------
        local : bool
            True if accessing data locally
        shot_ids : list[int]
            List of shots to use for PCA analysis 
        """

         # input
        config = self._load_config()

        if config is not None:
            # Extract parameters from config file
            max_components = config["max_components"]
            group = config["group"]
            signal_name = config["signal_name"]
            processes = config["processes"]
            nr_shots = config["nr_shots"]
        else:
            print("No configuration loaded")
            return

        # Use SLURM setting if available
        slurm_cpus = os.environ.get("SLURM_CPUS_PER_TASK")

        if slurm_cpus is not None:
            processes = int(slurm_cpus)
        else:
            # Fallback to user-provided or safe default
            if processes is not None and processes > 0:
                # Cap to system CPU count
                processes = min(processes, os.cpu_count() or 1)
            else:
                # Default to 1 if nothing was specified
                processes = 1
                
        # Instantiate MASTStorageManager
        store_manager = MASTStorageManager()

        # Process shot ids
        if nr_shots < len(shot_ids):
            shot_ids = shot_ids[:nr_shots]
       
        with Pool(processes=processes) as pool:
            args_iterable = [(  shot_id, 
                                group, 
                                signal_name, 
                                store_manager,
                                local) for shot_id in shot_ids]
            
            filter_none_results_iterator = (
                result for result in pool.imap(process_single_shot_id_star, args_iterable)
                if result is not None
            )
            results = list(tqdm(
                filter_none_results_iterator,
                total=len(shot_ids)
            ))

        if results:
            data_array = np.concatenate([*results], axis=1).T
        else:
            print("Warning: No results to concatenate.")

        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(data_array)

        # Determine the number of components to explain 
        # at least 99.7% of the variance
        for n_components in range(1,max_components):
            pca = PCA(n_components)
            pca.fit(scaled_data)

            explained_variance = pca.explained_variance_ratio_

            if np.cumsum(explained_variance)[-1] >= 0.997:
                break
        
        # Save pca model to output_dir 
        os.makedirs(output_directory, exist_ok=True)
        
        my_pca_file_name = output_directory+f"/pca_{group}_{signal_name}.joblib"
        joblib.dump({
            "pca": pca,
            "scaler": scaler,
            "group": group,
            "signal_name": signal_name
             },
            my_pca_file_name) 
        
    def __call__(self, local, shot_ids, output_directory):
        self.fit_PCA(local, shot_ids, output_directory)
    

def run_MASTpca(config_file, data_split_file, output_directory): 
    
    
    # Instance of PCAnalysis
    mast_pca = MASTpca(config_file)
    
    # Prepare sample of shot IDs
    shot_ids, _, _ = read_data_split_csv(csv_path = data_split_file)
    shot_ids = shuffle_shot_ids(shot_ids)
    
    # Fit pca to MAST shots
    local = True
    mast_pca(local, shot_ids, output_directory)
   
    
def test_reconstruction(
    shot_id, 
    local,
    pca_file_name,
    group,
    signal_name,  
    ):
    
    # Instantiate MASTStorageManager
    store_manager = MASTStorageManager()

    # Demonstrate PCA transform and reconstruction
    models = joblib.load(pca_file_name)
    pca = models["pca"]
    scaler = models["scaler"]

    vals = process_single_shot_id(
        shot_id,
        group,
        signal_name, 
        store_manager,
        local
        )

    if vals is None:
        return
    
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
    #plt.show() 
 
def test_run():
    # Input
    home_directory = "/home/ir-lore2"
    config_file = home_directory+"/fairmast-data-preprocessing/scripts/benchmarking/data_transform/config_files/config_pca_b_field_tor_probe_saddle_voltage.json"
    data_split_file = home_directory+"/fairmast-data-preprocessing/metadata/2025-05-12/data_splits.csv"
    output_directory = home_directory+"/fairmast-data-preprocessing/scripts/benchmarking/data_transform/data/output"
    run_MASTpca(config_file, data_split_file, output_directory)  
   
if __name__ == "__main__":
    shot_id = 16092
    local = True
    pca_file_name = "/home/ir-lore2/fairmast-data-preprocessing/scripts/benchmarking/data_transform/data/output/pca_magnetics_flux_loop_flux.joblib"
    group = "magnetics"
    signal_name = "flux_loop_flux"
    test_reconstruction(shot_id, local, pca_file_name, group, signal_name, )
    
    
    
    
    
    