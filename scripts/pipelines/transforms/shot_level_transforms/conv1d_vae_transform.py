import numpy as np
from collections import defaultdict
import torch

class Conv1dVAETransform:
    """
    Transform for conv1d-VAE training - creates signal segments for unsupervised learning.
    """

    def __call__(self, list_samples):
        """
        Transform windowed samples.

        Parameters
        ----------
        list_samples : list
        [
            {
                'x': {
                    signal_name1: {
                        'time': np.ndarray,   # shape (T_x,)
                        'values': np.ndarray  # shape (C, T_x)
                    },
                    signal_name2: {
                        'time': np.ndarray,   # shape (T_x,)
                        'values': np.ndarray  # shape (C, T_x)
                    },
                    …..
                },
                'y': {
                    signal_name1: {
                        'time': np.ndarray,   # shape (T_y,)
                        'values': np.ndarray  # shape (C, T_y)
                    },
                    signal_name2: {
                                'time': np.ndarray,   # shape (T_y,)
                                'values': np.ndarray  # shape (C, T_y)
                    },
                    ...
                },
            'window_index': int,  # The same index for x, y does not mean necessarily same time, it means that x and y are to be considered input-target pair
            }
            ... Same for a different 'window_index'
        ]

        Returns
        -------
        vae_samples : dict
            dict of signal_name: [signal_values1, ..., signal_values_idx].
            HINT: the length of the list is the same for all signals 
            in the same shot. However, it can vary across different shots.
        """
        all_signals = defaultdict(list)
        
        # Loop trhough all window_index
        for windowed_signal in list_samples:

            # Add signals
            for signal_name, signal_data in windowed_signal["x"].items():
                all_signals[signal_name].append(torch.tensor(signal_data["values"]))

        return all_signals