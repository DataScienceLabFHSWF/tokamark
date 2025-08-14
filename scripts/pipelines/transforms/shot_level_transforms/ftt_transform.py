"""
ft_transform_prep.py

Shot-level transform for preparing data from WindowSegmenterTransform
into the format expected by the FTTransformer pipeline.

This class is designed to be passed as the `shot_level_transform` in MastDataset.
It converts each (x, y) window into:

    Xs         : List[np.ndarray] of model input arrays (one per input modality).
    names      : List[str] active target names for this sample.
    y_native   : Dict[target_name] -> np.ndarray in native (d1, d2, d3) shape.

Unlike the PerTargetDataset approach, this transform directly outputs tuples
that can be batched with a matching collate function, avoiding an extra dataset wrapper.
"""

from typing import List, Dict, Any, Tuple
import numpy as np


class FTTransformPrep:
    """
    Prepare segmented windows into FTTransformer-ready tuples.

    Parameters
    ----------
    x_keys : List[str]
        List of signal names to be used as inputs (in order).
    y_key_to_target : Dict[str, str]
        Mapping from signal names in the window's 'y' dict to target names
        as used in the model's target registry.
    ensure_3d : bool, default=True
        If True, reshapes target arrays to (d1, d2, d3) by adding singleton dims
        where necessary.
    """

    def __init__(
        self,
        x_keys: List[str],
        y_key_to_target: Dict[str, str],
        ensure_3d: bool = True
    ):
        self.x_keys = x_keys
        self.y_key_to_target = y_key_to_target
        self.ensure_3d = ensure_3d

    def __call__(self, shot: Dict[str, Any]) -> List[Tuple[List[np.ndarray], List[str], Dict[str, np.ndarray]]]:
        """
        Convert a shot (already segmented into windows) into a list of FTTransformer tuples.

        Parameters
        ----------
        shot : dict
            Output from WindowSegmenterTransform: list of windows or a dict per window.

        Returns
        -------
        samples : List of tuples
            Each tuple is (Xs, names, y_native):
                Xs         : List[np.ndarray] inputs for the model
                names      : List[str] active target names
                y_native   : Dict[target_name] -> np.ndarray
        """
        if not isinstance(shot, list):
            raise ValueError("FTTransformPrep expects shot-level transform input to be a list of windows.")

        samples = []
        for window in shot:
            # 1. Collect X inputs in specified order
            Xs = []
            for key in self.x_keys:
                if key not in window["x"]:
                    raise KeyError(f"Missing expected x_key '{key}' in window.")
                arr = window["x"][key]["values"]
                Xs.append(np.asarray(arr))

            # 2. Collect targets
            y_native: Dict[str, np.ndarray] = {}
            names: List[str] = []
            for y_key, target_name in self.y_key_to_target.items():
                if y_key not in window["y"]:
                    raise KeyError(f"Missing expected y_key '{y_key}' in window.")
                arr = np.asarray(window["y"][y_key]["values"])
                if self.ensure_3d:
                    while arr.ndim < 3:
                        arr = np.expand_dims(arr, axis=-1)
                y_native[target_name] = arr
                names.append(target_name)

            samples.append((Xs, names, y_native))

        return samples
