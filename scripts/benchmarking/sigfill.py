"""
Classes for filling missing data in MAST signals.
"""
import matplotlib.pyplot as plt
import numpy as np
import os
import json
import pandas as pd
from sklearn.impute import SimpleImputer
import sys

cwd = os.path.dirname(os.path.abspath(__file__))
mother_dir = os.path.dirname(cwd) + os.sep
sys.path.append(os.path.abspath(os.path.join(mother_dir , "MAST_tools")))
sys.path.append(mother_dir)

from signal_utils import MASTSignalManager  
from store_utils import MASTStorageManager
from  utils import (is_finite_numeric_array,
                    read_signals,
                    make_dataframe_from_shot_ids,
                    shuffle_shot_ids)
 

class SimpleFiller:
    def __init__(self):
        pass

    def simple_fill(self, mast_signal, store):
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

def test_simple_filler(local, singularity, group, signal_name, shot_id):
    
    sig = MASTSignalManager(group, signal_name, shot_id)

    store_manager = MASTStorageManager(local_root_path =  "/srv")
    store= store_manager.make_shot_store(shot_id=shot_id, local=local, singularity=singularity)

    imputer = SimpleFiller()

    return imputer.simple_fill(sig, store)

class ConditionalFiller:

    def __init__(self, shot_tab_list, col_keys_list, col_means=None, col_covars=None, random=-1):
        self.shot_tab_list = shot_tab_list
        self.col_keys_list = col_keys_list
        self.col_means = col_means
        self.col_covars = col_covars
        if random < 0:
            self.deterministic = True
        else:
            self.deterministic = False
            np.random.seed(random)

    def subcov(self, col_covars, list_1, list_2):
        """Extract submatrix from full covariance matrix"""
        sub = np.array([[col_covars[i][j] for j in list_2] for i in list_1])
        return sub


    def stack_shots(self, shot_tab_list):
        return pd.concat(shot_tab_list, axis=0, ignore_index=True)

    def masked_mean(self, shot_tab_list, col_keys_list):
        all_data = self.stack_shots(shot_tab_list)
        return {col: all_data[col].mean(skipna=True) for col in col_keys_list}

    def masked_covar(self, shot_tab_list, col_keys_list):
        all_data = self.stack_shots(shot_tab_list)
        return all_data[col_keys_list].cov(min_periods=1).fillna(0).to_dict()

    def fit_mu_cov(self, shot_tab_list=None, col_keys_list=None, fit_mu=True, fit_cov=True):
        if shot_tab_list is None:
            shot_tab_list = self.shot_tab_list
        if col_keys_list is None:
            col_keys_list = self.col_keys_list
        else:
            self.col_keys_list = col_keys_list

        self.allshots_tab = self.stack_shots(shot_tab_list)

        col_means = self.masked_mean(shot_tab_list, col_keys_list)
        col_covars = self.masked_covar(shot_tab_list, col_keys_list)

        if fit_mu:
            self.col_means = col_means
        if fit_cov:
            self.col_covars = col_covars

    def fill_shots(self, shot_tab_list=None, col_keys_list=None, col_means=None, col_covars=None, missing_val=np.nan):
        if shot_tab_list is None:
            shot_tab_list = self.shot_tab_list
        if col_keys_list is None:
            col_keys_list = self.col_keys_list
        if col_means is None:
            col_means = self.col_means
        if col_covars is None:
            col_covars = self.col_covars

        filled_list = []

        for tshot in shot_tab_list:
            tshot_copy = tshot.copy()
            missing_cols = [col for col in col_keys_list if tshot[col].isna().any()]
            given_cols = [col for col in col_keys_list if not tshot[col].isna().any()]

            if len(missing_cols) ==0:
                filled_list.append(tshot_copy)
                continue

            # Build conditional distribution
            sigma_11 = self.subcov(col_covars, missing_cols, missing_cols)
            sigma_00 = self.subcov(col_covars, given_cols, given_cols)
            sigma_10 = self.subcov(col_covars, missing_cols, given_cols)

            X_given = tshot_copy[given_cols].to_numpy()
            mu_given = np.array([col_means[col] for col in given_cols])
            mu_missing = np.array([col_means[col] for col in missing_cols])

            dX_given = X_given - mu_given

            # Estimate conditional mean of missing variables
            X_missing_t = mu_missing.reshape(-1, 1) + sigma_10 @ np.linalg.pinv(sigma_00) @ dX_given.T

            if self.deterministic:
                X_missing_eps = np.zeros_like(X_missing_t)
            else:
                cov_post = sigma_11 - sigma_10 @ np.linalg.pinv(sigma_00) @ sigma_10.T
                X_missing_eps = np.random.multivariate_normal(np.zeros(len(missing_cols)), cov_post)

            X_missing = X_missing_t + X_missing_eps

            # Fill the missing values
            for idx, col in enumerate(missing_cols):
                tshot_copy[col] = X_missing[idx]

            filled_list.append(tshot_copy)

        return filled_list
    
        
    def conditional_fill(self):

        try:
            self.fit_mu_cov()
            filled_shots = self.fill_shots()
            return filled_shots

        except Exception as e:
            print(f"Exception {e}")
            return None
    
        
  
def test_conditional_fill(local, singularity, group, signal_name, shot_ids):
    store_manager = MASTStorageManager(local_root_path = "/srv")

    channels, df = make_dataframe_from_shot_ids(store_manager,
                    shot_ids,
                    group, 
                    signal_name,
                    local=local,
                    singularity=singularity)


    filler = ConditionalFiller(df, channels)
    filled_shots = filler.conditional_fill()
    
    return filled_shots


def signals_average_across_shots(store_manager : MASTStorageManager, 
                                 shot_ids : list[int],
                                 filepath:str,
                                 local : bool,
                                 singularity: bool):
    """
    For each shot:

    1- considers all signal vectors listed in the 
    filepath

    2- calculates the average value of each channel 
    in the signal. 

    3- Set to zero channels that contain NaN;

    3- returns the average value across shots in a json format:
    {
        "data": [
            {
                "signal_name": [ 
                    [avergae channel 0 ],
                    [average channel 1],
                    ...
                ],
                ...
            }
        ]
    }
    """

    # Read file containing signal names and the number of channels for each signal
    data_set = read_signals(filepath)

    data_set_counter = {}
    data_set_sums = {}

    # Loop through shots
    for nr,shot_id in enumerate(shot_ids[:5]):

        if nr>=0 :
            print(f"Shot number {nr+1} out of {len(shot_ids)}")
        store = store_manager.make_shot_store(shot_id, local=local, singularity = singularity)

        for key, value in data_set.items():
            # Retrieve signal group and name
            group, signal_name = key.split("/")

            # Form instance of SIGNAL
            sig = MASTSignalManager(group, signal_name, shot_id)

            # Prepare tracking
            data_set_counter.setdefault(signal_name, np.zeros(value))
            data_set_sums.setdefault(signal_name, None)

            try:
                vals = sig.get_values(store)

                mean_vals = np.mean(vals, axis=1) # Returns NaN if NaN is in vals

                # Check mean_vals are all numeric different from NaN and Inf
                for i, mean_val in enumerate(mean_vals):

                    if is_finite_numeric_array(mean_val):
                        data_set_counter[signal_name][i] += 1
                    else:
                        mean_vals[i]=0

                # Add mean_vals to data_set_sums
                if data_set_sums[signal_name] is None:
                    data_set_sums[signal_name] =  mean_vals
                else:
                    data_set_sums[signal_name] += mean_vals

            except Exception as e:
                print(f"Exception: {e}")

    # Calculate average 
    data = []
    for signal_name, summed_array in data_set_sums.items():
        
        counter = data_set_counter[signal_name]
        
        averaged_channels = []
        for channel, count in zip(summed_array, counter):
            if channel and count> 0:
                averaged_channels.append(channel / count)
            else:
                print(f"No valid data to average for value '{signal_name}'.")

        data.append({signal_name: averaged_channels})
    
    json_data = {"data":data}
    with open("averaged_arrays.json", "w") as f:
        json.dump(json_data, f, indent=4)


def test_signals_average_across_shots():

    # Get all shot ids
    local = False
    singularity = True
    store_manager = MASTStorageManager(local_root_path = "/srv")
    shot_ids = store_manager.list_all_shots(local= local, singularity = singularity)
    signals_average_across_shots(store_manager,
                                shot_ids[:5], 
                                "data/list_of_signals.txt", 
                                local,
                                singularity)




if __name__ == "__main__":

    # input
    local = False
    singularity = True
    group = "magnetics"
    signal_name = "flux_loop_flux"
    shot_id = 16092

    # MAST store manager
    store_manager = MASTStorageManager(local_root_path =  "/srv")
    shot_ids = store_manager.list_all_shots(local=local, singularity = singularity)
    shot_ids = shuffle_shot_ids(shot_ids)[:6]

    # Re-order shot_ids so that shot_id is always first
    new_shot_ids = shot_ids[:]
    if shot_id in new_shot_ids:
        shot_ids.remove(shot_id)
        new_shot_ids.insert(0, shot_id)
    else:
        new_shot_ids.insert(0, shot_id)
    shot_ids = new_shot_ids

    # Impute missing channels for test shot_id
    conditional_filled = test_conditional_fill(local, singularity, group, signal_name, shot_ids)[0].T
    conditional_filled = conditional_filled.to_numpy()
    simple_filled = test_simple_filler(local, singularity, group, signal_name, shot_id)

    # Retrieve original signal
    sig = MASTSignalManager(group, signal_name, shot_id)
    store = store_manager.make_shot_store(shot_id=shot_id, local=local, singularity=singularity)
    vals = sig.get_values(store)

    # Plot
    fig, axes = plt.subplots(nrows=3, figsize=(8, 10))

    vmin = min(conditional_filled.min(), simple_filled.min(), vals.min())
    vmax = max(conditional_filled.max(), simple_filled.max(), vals.max())

    p0 = axes[0].pcolorfast(vals, vmin=vmin, vmax=vmax)
    p1 = axes[1].pcolorfast(conditional_filled, vmin=vmin, vmax=vmax)
    p2 = axes[2].pcolorfast(simple_filled, vmin=vmin, vmax=vmax)


    axes[0].set_title(r"$\bf{Original}$" + f" {group}/{signal_name} shot id = {shot_ids[0]}")
    fig.colorbar(p0, ax=axes[0])

    axes[1].set_title(r"$\bf{Conditional fill}$")
    fig.colorbar(p1, ax=axes[1])

    axes[2].set_title(r"$\bf{Simple fill}$")
    fig.colorbar(p2, ax=axes[2])

    plt.tight_layout()

    fig.savefig(f"{group}_{signal_name}.png", bbox_inches="tight", dpi=300)
    plt.show()

