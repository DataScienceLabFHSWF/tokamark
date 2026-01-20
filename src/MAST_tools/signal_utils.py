"""
Docstring reference: https://numpydoc.readthedocs.io/en/latest/format.html
Python style reference: https://google.github.io/styleguide/pyguide.html
"""

import time
import numpy as np
import xarray as xr
from typing import Union, Optional, Any
from types import NoneType

try:
    from . import store_utils
    from . import constants as cc
except ImportError:
    import store_utils
    import constants as cc


# ======================================================================================================================
class MASTSignalManager:
    """
    Class with signal management tools for MAST data.

    Attributes
    ----------
    signal_manager_id : str
        User-defined signal manager ID.
    store_manager_settings : Optional[dict]
        Settings for the store manager instance.
    store_manager : store_utils.MASTStorageManager
        Store manager instance.

    Methods
    -------
    _set_store_manager(store_manager)
        Set the store_manager instance attribute.
    get_source_profiles(data_origin, source_name)
        Get source profiles from a given data origin.
    get_signal_values(signal_name, data_origin, source_name, verbose)
        Get signal values from a given data origin.
    get_signal_times_and_time_type(signal_name, data_origin, source_name, verbose)
        Get signal times and time type from a given data origin.
    get_signal_profile(signal_name, data_origin, source_name, verbose)
        Get signal profile from a given data origin.
    get_channel_names(signal_name, data_origin, source_name, verbose)
        Get channel names from a given data origin.

    """

    # ------------------------------------------------------------------------------------------------------------------
    def __init__(
            self,
            manager_id: str = "",
            store_manager_settings: Optional[dict] = None
    ) -> None:
        """
        Initialise class attributes.

        Parameters
        ----------
        manager_id : str
            User-defined manager ID. Default: "".
        store_manager_settings : Optional[dict]
            Settings for the store manager instance. If None, a generic store manage instance is created with default
            values as defined in the docstrings of `store_utils.MASTStorageManager`.
            Optional. Default: None.

        Raises
        ------
        TypeError
            If parameter `store_manager_settings` is not a dict.

        Returns
        -------
        None

        Notes
        -----
        - Upon creation of a signal manager instance from the `MASTSignalManager` class, a store manager from the
          `MASTStorageManager` class is created as instance attribute. This facilitates the process to get signal values
          as no separate store manager instance must be created. However, if an existing store manager instance is
          available, it could be passed.

        """

        self.signal_manager_id = manager_id

        if store_manager_settings is None:
            self.store_manager_settings = {}  # I.e., use default values in `store_utils.MASTStorageManager.__init__`.
        else:
            self.store_manager_settings = store_manager_settings
        self.store_manager = store_utils.MASTStorageManager(**self.store_manager_settings)

    # ------------------------------------------------------------------------------------------------------------------
    def _set_store_manager(
            self,
            store_manager: store_utils.MASTStorageManager
    ) -> None:
        """
        Set the `store_manager` attribute.

        Parameters
        ----------
        store_manager : store_utils.MASTStorageManager
            Instance of store_utils.MASTStorageManager class.

        Returns
        -------
        None

        Raises
        ------
        None

        """

        self.store_manager = store_manager

    # ------------------------------------------------------------------------------------------------------------------
    def get_source_profiles(
           self,
           data_origin: Union[dict, cc.ZarrStoreType],
           source_name: str,
           verbose: bool = False
    ) -> xr.Dataset:
        """
        Get source profiles from a given data origin.

        Parameters
        ----------
        data_origin : Union[dict, cc.ZarrStoreType]
            Origin of data for source profile creation.
        source_name : str
            Name of target source.

        Returns
        -------
        xr.Dataset
            Source profiles from given data origin.

        Raises
        ------
        None

        """

        self.store_manager._check_data_origin(data_origin)  # noqa

        if isinstance(data_origin, dict):
            # From shot info
            store = self.store_manager.make_shot_store(shot_info=data_origin)
        else:
            # From store
            store = data_origin

        source_profile = None
        try:
            source_profile = xr.open_zarr(store=store, group=source_name)
        except KeyError as e:
            if verbose:
                print(f"Exception: {e}")

        return source_profile

    # ------------------------------------------------------------------------------------------------------------------
    def get_signal_values(
            self,
            signal_name: str,
            data_origin: Union[dict, cc.ZarrStoreType, cc.XarrayDatasetType],
            source_name: Optional[str] = None,
            verbose: bool = False
    ) -> Union[np.ndarray, None]:
        """
        Get signal values from a given data origin.

        Parameters
        ----------
        signal_name : str
            Name of the target signal.
        data_origin : Union[dict, cc.ZarrStoreType, cc.XarrayDatasetType]
            Origin of data for signal value retrieval.
        source_name : Optional[str]
            Name of target source. If `data_origin` is a Zarr store, `source_name` must be provided.
            Optional. Default: None.
        verbose : bool
            If True, verbose mode is activated.
            Default: False.

        Returns
        -------
        Union[numpy.ndarray, None]
            Signal values, or None if error.

        Raises
        ------
        None

        """

        signal_profile = self.get_signal_profile(
            signal_name=signal_name,
            data_origin=data_origin,
            source_name=source_name,
            verbose=verbose
        )

        if signal_profile is not None:
            return signal_profile.values
        else:
            # If here, an error occurred while creating the signal profile.
            return None

    # ------------------------------------------------------------------------------------------------------------------
    def get_signal_times_and_time_type(
            self,
            signal_name: str,
            data_origin: Union[dict, cc.ZarrStoreType, cc.XarrayDatasetType],
            source_name: Optional[str] = None,
            verbose: bool = False
    ) -> Union[tuple[np.ndarray, str], tuple[None, None]]:
        """
        Get signal times and time type from a given data origin.

        Parameters
        ----------
        signal_name : str
            Name of the target signal.
        data_origin : Union[dict, cc.ZarrStoreType, cc.XarrayDatasetType]
            Origin of data for signal value retrieval.
        source_name : Optional[str]
            Name of target source. If `data_origin` is a Zarr store, `source_name` must be provided.
            Optional. Default: None.
        verbose : bool
            If True, verbose mode is activated.
            Default: False.

        Returns
        -------
        Union[list[np.ndarray, str], list[None, None]]
            List with signal times with time type (i.e., list[np.ndarray, str]), or [None, None] if error.

        Raises
        ------
        None

        """

        signal_profile = self.get_signal_profile(
            signal_name=signal_name,
            data_origin=data_origin,
            source_name=source_name,
            verbose=verbose
        )

        if signal_profile is not None:

            try:
                time_type_ = [str(kk) for kk in signal_profile.coords.keys() if str(kk).startswith("time")][0]
            except IndexError:
                # If here, no time info was found, and so signal is not time-dependent.
                return None, None

            signal_times_ = signal_profile[time_type_].values

            return signal_times_, time_type_
        else:
            # If here, an error occurred while creating the signal profile.
            return None, None

    # ------------------------------------------------------------------------------------------------------------------
    def get_signal_profile(
            self,
            signal_name: str,
            data_origin: Union[dict, cc.ZarrStoreType, cc.XarrayDatasetType],
            source_name: Optional[str] = None,
            verbose: bool = False
    ) -> Union[xr.DataArray, NoneType, Any]:
        """
        Get signal profile from a given data origin.

        Parameters
        ----------
        signal_name : str
            Name of the target signal.
        data_origin : Union[dict, cc.ZarrStoreType, cc.XarrayDatasetType]
            Origin of data for signal value retrieval.
        source_name : Optional[str]
            Name of target source. If `data_origin` is a Zarr store, `source_name` must be provided.
            Optional. Default: None.
        verbose : bool
            If True, verbose mode is activated.
            Default: False.

        Returns
        -------
        Union[xr.DataArray, None]
            Signal profile, or None if error.

        Raises
        ------
        None

        """

        if isinstance(data_origin, cc.XarrayDatasetType):
            # From group profile (i.e., xarray.core.dataset)
            profile = data_origin

        else:
            # From shot info or cc.ZarrStoreType

            try:
                self.store_manager._check_data_origin(data_origin)  # noqa
            except Exception as e:
                if verbose:
                    print(f"Exception: {e}")

            if isinstance(data_origin, dict):
                # From shot info
                store = self.store_manager.make_shot_store(shot_info=data_origin)
            else:
                # From cc.ZarrStoreType (cc.ZarrFSStoreType or cc.ZarrLocalStoreType)
                store = data_origin

            profile = None
            try:
                profile = xr.open_zarr(store=store, group=source_name)
            except KeyError as e:
                if verbose:
                    print(f"Exception: {e}")

        if profile is not None:
            try:
                return profile[signal_name]
            except KeyError:
                if verbose:
                    print(f"Invalid 'signal_name' {signal_name}.")
                return None
        else:
            # If here, an error occurred while creating the signal profile.
            return None

    # ------------------------------------------------------------------------------------------------------------------
    def get_channel_names(
            self,
            signal_name: str,
            data_origin: Union[dict, cc.ZarrStoreType, cc.XarrayDatasetType],
            source_name: Optional[str] = None,
            verbose: bool = False
    ) -> Union[np.ndarray, None]:
        """
        Get signal channel names.

        Parameters
        ----------
        signal_name : str
            Name of the target signal.
        data_origin : Union[dict, cc.ZarrStoreType, cc.XarrayDatasetType]
            Origin of data for signal profile retrieval (e.g., a Zarr store).
        source_name : Optional[str]
            Name of target source. If `data_origin` is a Zarr store, `source_name` must be provided.
            Optional. Default: None.
        verbose : bool
            If True, verbose mode is activated.
            Default: False.

        Returns
        -------
        Union[np.ndarray, None]
            Available signal channels as np.ndarray, or None.

        Raises
        ------
        None

        """

        try:
            signal_profile = self.get_signal_profile(
                signal_name=signal_name,
                data_origin=data_origin,
                source_name=source_name,
                verbose=verbose
            )

            non_time_coords = [coord for coord in signal_profile.coords if coord != "time"]
            if non_time_coords:
                channel_coord = non_time_coords[0]
                return signal_profile.coords[channel_coord].values
            else:
                return None
        except Exception as e:
            if verbose:
                print(f"Error opening Zarr store: {e}")
            return None


# ----------------------------------------------------------------------------------------------------------------------
def test() -> None:
    """
    Quick tests for module functionality.

    Return
    ------
    None

    """

    t0_all_tests = time.time()

    # ..................................................................................................................

    TESTS_TO_RUN = {  # noqa
        "source_from_store": True,
        "signal_values_from_store": True,
        "signal_values_from_shot_info": True,
        "signal_times_from_shot_info": True
    }

    signal_manager = MASTSignalManager()
    # print(type(signal_manager.store_manager))

    # ..................................................................................................................

    shot_info = {"shot_id": 30421, "level": 2, "test_data": False, "local": False, "via_parquet": False}
    source_name = "magnetics"
    signal_name = "flux_loop_flux"

    # ..................................................................................................................
    # Get signal values from store
    if TESTS_TO_RUN["source_from_store"]:

        source_from_shot_info = signal_manager.get_source_profiles(
            data_origin=shot_info,
            source_name=source_name
        )

        print(source_from_shot_info)
        print(type(source_from_shot_info))

        print(source_from_shot_info[signal_name])
        print(type(source_from_shot_info[signal_name]))

    # ..................................................................................................................
    # Get signal values from store
    if TESTS_TO_RUN["signal_values_from_store"]:

        store_from_shot_info = signal_manager.store_manager.make_shot_store(shot_info=shot_info)
        signal_values = signal_manager.get_signal_values(
            signal_name=signal_name,
            data_origin=store_from_shot_info,
            source_name=source_name,
            verbose=True
        )

        print(f"Signal values: {signal_values}\n")

    # ..................................................................................................................
    # Get signal values from shot info

    if TESTS_TO_RUN["signal_values_from_shot_info"]:

        signal_values = signal_manager.get_signal_values(
            signal_name=signal_name,
            data_origin=shot_info,
            source_name=source_name
        )

        print(f"Signal values: {signal_values}\n")

    # ..................................................................................................................
    # Get signal times from shot info

    if TESTS_TO_RUN["signal_times_from_shot_info"]:

        signal_times, signal_type = signal_manager.get_signal_times_and_time_type(
            signal_name=signal_name,
            data_origin=shot_info,
            source_name=source_name
        )

        print(f"Signal type: '{signal_type}'\n")
        print(f"Signal times: {signal_times}\n")

    # ..................................................................................................................

    print("---------------------------------------------")
    print(f"Elapsed time for tests() execution: {round(time.time() - t0_all_tests, 2)} s")


# ======================================================================================================================
if __name__ == "__main__":
    test()
