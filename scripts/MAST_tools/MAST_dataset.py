"""
Docstring reference: https://numpydoc.readthedocs.io/en/latest/format.html
Python style reference: https://google.github.io/styleguide/pyguide.html
"""

import time
import numpy as np
from pprint import pprint
from typing import Union, Callable, Optional
from torch.utils.data import Dataset
try:
    from . import signal_utils
    from . import constants as cc
except ImportError:
    import signal_utils
    import constants as cc


# ======================================================================================================================
class MastDataset(Dataset):
    """
    Dataset class for MAST data.

    If `return_incomplete_shot` is True, `__getitem__` returns shots/windows even when some variables are missing
    (time/values is `None`). This lets downstream windowing mark missing inputs per-sample with None.

    Attributes
    ----------
    local : bool
        Boolean flag to activate local mode. If True, use local MAST database, otherwise use remote S3 bucket.
    level : int
        Target level for the MAST data/metadata to be pulled.
    test_data : bool
        If True, the target shot is pulled from test data, otherwise it is pulled from curated data. Not available
        for locally stored data (i.e, if `local` is True).
    shots_list : list[int]
        List of shot IDs.
    source_signal_list : list[list[str]]
        List of data names to load using the format [[<source>, <signal>], ..., [<source>, <signal>]].
    signal_level_transform_map : Optional[dict[str, Callable]]
        Map of transforms to apply at signal level.
    shot_level_transform : Optional[Callable]
        Transform to apply at shot level.
    return_incomplete_shots : bool
        Boolean flag to allow retrieval of incomplete shots.
    sig : signal_utils.MASTSignalManager
        Instance of `signal_utils.MASTSignalManager`.
    verbose : bool
        Boolean flag to activate/deactivate verbose mode.

    Methods
    -------
    __len__
        Return the size of the dataset.
    __getitem__(idx)
        Return samples by shot index.
    get_shot_id(idx)
        Return shot ID from shot index.
    get_windows_for_shot(idx)
        Return the list of windows for the given shot index ([] if empty/missing).

    """

    # ------------------------------------------------------------------------------------------------------------------
    def __init__(
        self,
        local: bool,
        shots_list: list[int],
        source_signal_list: list[list[str]],
        signal_level_transform_map: Optional[dict[str, Callable]] = None,
        shot_level_transform: Optional[Callable] = None,
        return_incomplete_shots: bool = False,
        store_manager_settings: Optional[dict] = None,
        other_mast_settings: Optional[dict] = None,
        verbose: bool = False
    ) -> None:
        """
        Initialise class attributes.

        Parameters
        ----------
        local : bool
            If True, use local MAST database, otherwise use remote S3 bucket.
        shots_list : list[int]
            List of shot IDs to load data for.
        source_signal_list : list[list[str]]
            List of data names to load using the format [[<source>, <signal>], ..., [<source>, <signal>]].
        signal_level_transform_map : Optional[dict[str, Callable]]
            Map of transforms to apply at signal level.
            Optional. Default: None.
        shot_level_transform : Optional[Callable]
            Transform to apply at shot level.
            Optional. Default: None.
        return_incomplete_shots : bool
            If True, DO NOT drop shots with missing variables, and pass them through so that the windowing transform can
            insert None for missing inputs per window.
            Optional. Default: False.
        store_manager_settings : Optional[dict]
            Settings for the store manager instance. If None, a generic store manage instance is created with default
            values as defined in the docstrings of `store_utils.MASTStorageManager`.
            Optional. Default: None.
        other_mast_settings : Optional[dict]
            Other MAST-related settings, provided via a dictionary with suitable (key, value) pairs. Valid settings are:
            level : int
                Target level for the MAST data/metadata to be pulled.
            test_data : bool
                If True, the target shot is pulled from test data, otherwise it is pulled from curated data. Not
                available for locally stored data (i.e, if `local` is True).
            Optional. Default: None, which results in level = 2 and test_data = False.
        verbose : bool
            If True, verbose mode is activated.
            Optional. Default: False.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If value of field 'level' in 'other_mast_settings' is not in [1, 2].
        TypeError
            If value of field 'level' in 'other_mast_settings' is not of type 'int'.
        TypeError
            If value of field 'test_data' in 'other_mast_settings' is not of type boolean.

        """

        # General MAST settings
        if other_mast_settings:
            self.local = local
            if "level" in other_mast_settings:
                if isinstance(other_mast_settings["level"], int):
                    if other_mast_settings["level"] in [1, 2]:
                        self.level = other_mast_settings["level"]
                    else:
                        raise TypeError(
                            "Value of field 'level' in 'other_mast_settings' must be of type 'int'."
                        )
                else:
                    raise ValueError(
                        "Value of field 'level' in 'other_mast_settings' must be in [1, 2]."
                    )
            else:
                self.level = 2

            if "test_data" in other_mast_settings:
                if isinstance(other_mast_settings["test_data"], bool):
                    self.test_data = other_mast_settings["test_data"]
                else:
                    raise TypeError(
                        "Value of field 'test_data' in 'other_mast_settings' must be of type boolean."
                    )
            else:
                self.test_data = False
        else:
            self.local = local
            self.level = 2
            self.test_data = False

        # Shot-specific settings
        self.shots_list = shots_list
        self.source_signal_list = source_signal_list
        self.signal_level_transform_map = signal_level_transform_map
        self.shot_level_transform = shot_level_transform
        self.return_incomplete_shots = return_incomplete_shots

        # Signal manager instance
        self.sig = signal_utils.MASTSignalManager(store_manager_settings=store_manager_settings)

        # Other settings
        self.verbose = verbose

    # ------------------------------------------------------------------------------------------------------------------
    def __len__(
            self
    ) -> int:
        """
        Return the size of the dataset.

        Returns
        -------
        int
            Length of shot_list.

        """

        return len(self.shots_list)

    # ------------------------------------------------------------------------------------------------------------------
    def __getitem__(
            self,
            idx: int
    ) -> Union[list, dict]:
        """
        Return samples by shot index.

        Parameters
        ----------
        idx : int
            Shot index.

        Returns
        -------
        Union[list, dict]
            Either a list of chunks, or a given the raw shot dict.

        """

        store_manager = self.sig.store_manager
        store = store_manager.make_shot_store(
            shot_info={
                "shot_id": self.shots_list[idx],
                "level": self.level,
                "test_data": self.test_data,
                "local": self.local
            },
            verbose=self.verbose
        )

        shot = {}

        # Collect variables (i.e. source-signal) of interest
        for source, signal in self.source_signal_list:
            shot_profile = self.sig.get_signal_profile(
                data_origin=store,
                source_name=source,
                signal_name=signal,
                verbose=self.verbose
            )

            if shot_profile is not None:
                try:
                    shot_time, _ = self.sig.get_signal_times_and_time_type(
                        signal_name=signal,
                        data_origin=store,
                        source_name=source,
                        verbose=self.verbose
                    )
                except Exception as e:
                    print(f"Error getting time for shot {self.shots_list[idx]}: {e}")
                    shot_time = None
                try:
                    shot_vals = (
                        np.expand_dims(shot_profile.values, axis=0)
                        if shot_profile.values.ndim == 1
                        else shot_profile.values
                    )
                except AttributeError:
                    shot_vals = None
            else:
                shot_vals = None
                shot_time = None

            # Apply variable-level transforms only if we have both time and values
            if self.signal_level_transform_map and (shot_vals is not None and shot_time is not None):
                shot[f"{source}-{signal}"] = self.signal_level_transform_map[f"{source}-{signal}"](
                    {"time": shot_time, "values": shot_vals}
                )
            else:
                # Keep missing signals as {"time": np.array([]), "values": np.array([])}
                shot[f"{source}-{signal}"] = {"time": np.array([]), "values": np.array([])}

        # Apply shot-level transforms to obtain a list of training objects (windows)
        if self.shot_level_transform:
            if self.return_incomplete_shots:
                # Pass through even if some variables are missing;
                item = self.shot_level_transform(shot)
                return item if isinstance(item, list) else [item]
            else:
                # Legacy behavior: drop shots with any missing variable
                if all(subval is not None for subdict in shot.values() for subval in subdict.values()):
                    item = self.shot_level_transform(shot)
                    return item if isinstance(item, list) else [item]
                else:
                    return []
        else:
            # No shot-level transform → return the raw shot dict (may include None fields)
            return shot

    # ------------------------------------------------------------------------------------------------------------------
    class CachedDataset(Dataset):
        """ Cache base dataset into local memory

        Parameters
        ----------
        Dataset : PyTorch Dataset
        """
        def __init__(self, base_dataset):
            """
            base_dataset: PyTorch Dataset
            """
            self.base_dataset = base_dataset
            self.cache = [None] * len(base_dataset)
            self._is_cached = [False] * len(base_dataset)
        
        def __getitem__(self, idx):
            # If not cached, load it once
            if not self._is_cached[idx]:
                item = self.base_dataset[idx]

                self.cache[idx] = item
                self._is_cached[idx] = True

            return self.cache[idx]

        def __len__(self):
            return len(self.base_dataset)

    # ------------------------------------------------------------------------------------------------------------------
    def get_shot_id(
            self,
            idx: int
    ) -> int:
        """
        Return shot ID from shot index.

        Parameters
        ----------
        idx : int
            Shot index.

        Returns
        -------
        int
            Shot ID.

        Raises
        ------
        None

        """

        return self.shots_list[idx]

    # ------------------------------------------------------------------------------------------------------------------
    def get_windows_for_shot(
            self,
            idx: int
    ) -> list:
        """
        Return the list of windows for the given shot index ([] if empty/missing).

        Parameters
        ----------
        idx : int
            Shot ID.

        Returns
        -------
        list
            The list of windows for the given shot index.

        """

        try:
            obj = self.__getitem__(idx)
        except IndexError:
            return []

        if isinstance(obj, list):
            return obj
        else:
            # If here, 'obj' is the entire shot with no shot-level transform → treat as a single-window shot.
            return [obj]


# ----------------------------------------------------------------------------------------------------------------------
def tests() -> None:
    """
    Quick tests for module functionality.

    Return
    ------
    None

    """

    t0_all_tests = time.time()

    # ..................................................................................................................

    dummy_dataset = MastDataset(
        local=False,
        shots_list=[30421],
        source_signal_list=[["summary", "power_nbi"], ["magnetics", "b_field_pol_probe_obr_field"]],
        signal_level_transform_map=None,
        shot_level_transform=None,
    )

    print(f"\ndummy_dataset.__len__: {dummy_dataset.__len__()}")
    print(f"\ndummy_dataset.get_shot_id(0): {dummy_dataset.get_shot_id(0)}")

    print("\ndummy_dataset.__getitem__(0):")
    pprint(dummy_dataset.__getitem__(0))

    print("\ndummy_dataset.get_windows_for_shot(0):")
    pprint(dummy_dataset.get_windows_for_shot(0))

    # ..................................................................................................................

    print("---------------------------------------------")
    print(f"Elapsed time for tests() execution: {round(time.time() - t0_all_tests, 2)} s")


# ======================================================================================================================
if __name__ == "__main__":
    tests()
