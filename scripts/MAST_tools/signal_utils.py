# Contributors:
# - Rodrigo Ordonez-Hurtado (rodrigo.ordonez.hurtado@ibm.com).
# Remarks:
# - Based on previous implementation.
# Docstring reference: https://numpydoc.readthedocs.io/en/latest/format.html

from typing import Union
import xarray as xr
try:
    from . import store_utils
    from . import constants as cc
except ImportError:
    import store_utils
    import constants as cc


# ======================================================================================================================
class MASTSignalManager:
    """
    Class to process signals from fair MAST bucket
    """

    # ------------------------------------------------------------------------------------------------------------------
    def __init__(
            self,
            manager_id: str = "",
            store_manager_settings: Union[dict, cc.NoneType] = None
    ):
        """
        Attributes
        ----------
        manager_id : str
            User defined manager ID. Default: "".
        store_manager_settings : Union[dict, None]
            Settings for the store manager instance. If None, a generic store manage instance is created with default
            values as defined in the docstrings of store_utils.MASTStorageManager.
            Optional. Default: None.

        Methods
        -------
        get_source_profiles(data_origin, source_name)
            Get source profiles from a given data origin.
        get_signal_values(data_origin, source_name, signal_name)
            Get signal values from a given data origin.
        _set_store_manager(store_manager)
            Not yet implemented.

        Remarks
        -------
        - Upon creation of a signal manager instance from the MASTSignalManager class, a store manager (from the
          MASTStorageManager class is created as instance attribute. This facilitates the process to get signal values,
          as no separate store manager instance must be created. However, if an existing store manager instance is
          available, it could be passed

        """

        self.signal_manager_id = manager_id

        if store_manager_settings is None:
            self.store_manager_settings = {}
        else:
            assert isinstance(store_manager_settings, dict), "Type error: invalid store_manager_settings. It must be" \
                                                             " of type dict."
            self.store_manager_settings = store_manager_settings
        self.store_manager = store_utils.MASTStorageManager(**self.store_manager_settings)

    # ------------------------------------------------------------------------------------------------------------------
    def _set_store_manager(self, store_manager):
        raise NotImplementedError

    # ------------------------------------------------------------------------------------------------------------------
    def get_source_profiles(
           self,
           data_origin: Union[dict, cc.ZarrStoreType],
           source_name: str
    ):
        """
        Get source profiles from a given data origin.

        Parameters
        ----------
        data_origin : Union[dict, ZarrStoreType]
            Origin of data for source profile creation.
        source_name : str
            Name of target source.

        Returns
        -------
        xarray.core.dataset.Dataset
            Source profiles from given data origin.

        """
        self.store_manager._check_data_origin(data_origin)  # noqa.
        assert isinstance(source_name, str), "Type error: invalid source_name. It must be of type str."

        if isinstance(data_origin, dict):
            # From shot info
            store = self.store_manager.make_shot_store(shot_info=data_origin)
        else:
            # From store
            store = data_origin

        return xr.open_zarr(store=store, group=source_name)

    # ------------------------------------------------------------------------------------------------------------------
    def get_signal_values(
            self,
            signal_name: str,
            data_origin: Union[dict, cc.ZarrStoreType, cc.XarrayDatasetType],
            source_name: Union[str, cc.NoneType] = None
    ):
        """
        Get signal values from a given data origin.

        Parameters
        ----------
        data_origin : Union[dict, ZarrStoreType, cc.XarrayDatasetType]
            Origin of data for signal value retrieval.
        source_name : str
            Name of target source.
            Optional. Default: None.
        signal_name : str
            Name of target signal.
            Optional. Default: None.

        Returns
        -------
        numpy.ndarray
            Signal values.

        """
        assert isinstance(signal_name, str), "Type error: invalid source_name. It must be of type str."

        if isinstance(data_origin, cc.XarrayDatasetType):
            # From group profile (i.e., xarray.core.dataset)
            profile = data_origin

        else:
            self.store_manager._check_data_origin(data_origin)  # noqa.
            assert isinstance(source_name, str), "Type error: invalid source_name. It must be of type str."

            if isinstance(data_origin, dict):
                # From shot info
                store = self.store_manager.make_shot_store(shot_info=data_origin)

            else:
                # From store
                store = data_origin

            profile = xr.open_zarr(store=store, group=source_name)

        return profile[signal_name].values


# ----------------------------------------------------------------------------------------------------------------------
def test():

    signal_manager = MASTSignalManager()

    # ..................................................................................................................

    shot_info = {"shot_id": 30421, "level": 2, "test_data": False, "local": False, "via_parquet": False}
    source_name = "magnetics"
    signal_name = "flux_loop_flux"

    # ..................................................................................................................
    # Get signal values from store

    store_from_shot_info = signal_manager.store_manager.make_shot_store(shot_info=shot_info)
    signal_values = signal_manager.get_signal_values(
        signal_name=signal_name,
        data_origin=store_from_shot_info,
        source_name=source_name
    )

    print(f"Signal values: {signal_values}\n")

    # ..................................................................................................................
    # Get signal values from shot ID

    signal_values = signal_manager.get_signal_values(
        signal_name=signal_name,
        data_origin=shot_info,
        source_name=source_name
    )

    print(f"Signal values: {signal_values}\n")


# ======================================================================================================================
if __name__ == "__main__":
    test()
