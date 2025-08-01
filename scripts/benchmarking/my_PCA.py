import joblib
import json
from multiprocessing import Pool
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import os

import sys
cwd = os.path.dirname(os.path.abspath(__file__))
mother_dir = os.path.dirname(cwd) + os.sep
sys.path.append(os.path.abspath(os.path.join(mother_dir , "MAST_tools")))
sys.path.append(mother_dir)

from sigfill import run_simple_filler
from store_utils import MASTStorageManager


def process_single_shot_id( 
    shot_id:int,
    group:str,
    signal_name:str, 
    store_manager: MASTStorageManager,
    local: bool
    ):
    
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


class my_PCA:
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
        random_shot_ids: list[int]
        ):
        """Aply PCA 

        Parameters
        ----------
        local : bool
            True if accessing data locally
        random_shot_ids : list[int]
            List of shots to use for PCA analysis
            
        """

         # input
        config = self._load_config()

        if config:
            # Extract parameters from config file
            max_components = config["max_components"]
            group = config["group"]
            signal_name = config["signal_name"]
            processes = config["processes"]
        else:
            print("No configuration loaded")
            return

        # Instanciate MASTStorageManager
        store_manager = MASTStorageManager()

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

