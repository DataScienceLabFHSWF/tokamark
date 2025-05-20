# @Author: Rodrigo Ordonez-Hurtado (rodrigo.ordonez.hurtado@ibm.com)

import zarr
import zarr.storage
import fsspec
import s3fs
import numpy as np
import pandas as pd
import xarray as xr
import warnings
import matplotlib.pyplot as plt
from typing import Union
import logging

NoneType = type(None)
ZarrStoreType = zarr.storage.FsspecStore
logging.getLogger('asyncio').setLevel(logging.CRITICAL)


# ======================================================================================================================
class MASTStorageManager:
    def __init__(
            self,
            mast_app_url: str = "https://mastapp.site",
            base_fsspec_protocol: str = "simplecache",
            target_fsspec_protocol: str = "s3",
            s3_endpoint_url: str = "https://s3.echo.stfc.ac.uk",
            local_root_path: str = "/rds/project/rds-mOlK9qn0PlQ/fairmast/upload-tmp",
    ):
        """
        Attributes
        ----------
        mast_app_url : str
            MastApp URL to form the target file path used by the pd.read_parquet method.

        base_fsspec_protocol: str
            Base protocol used by fsspec. Some supported protocols include:
            "blockcache":
                - With this option, data is downloaded block-wise.
                - Restrictions:
                    - It has a storage/OS combination which supports sparse files.
                    - The backend implementation uses files which derive from AbstractBufferedFile.
                    - The library you pass the resultant object to accepts generic python file-like objects.
            "filecache":
                - Works for all file system implementations, and provides a real local file for other libraries to use.
            "simplecache":
                - Same as “filecache”, except without options for cache expiry and to check original source.
                - Only option guaranteed to be thread/process-safe.
            Full list of supported protocols obtained via fsspec.available_protocols().
            More info available at https://filesystem-spec.readthedocs.io/en/latest/features.html.
            Default: "simplecache".

        target_fsspec_protocol : str
            Target filesystem protocol for the selected base fsspec protocol.
            Default: "s3".

        s3_endpoint_url : str
            Endpoint of the cloud S3 bucket used for remote data pulling (i.e., for local=False).
            Default: "https://s3.echo.stfc.ac.uk".

        local_root_path : str
            Local root path used for local data pulling (i.e., for local=True).
            Default: "/rds/project/rds-mOlK9qn0PlQ/fairmast/upload-tmp", corresponding to the CSD3 cluster.

        """

        self.mast_app_url = mast_app_url
        self.base_fsspec_protocol = base_fsspec_protocol
        self.target_fsspec_protocol = target_fsspec_protocol
        self.s3_endpoint_url = s3_endpoint_url
        self.local_root_path = local_root_path

        self.fs_local_fsspec = fsspec.filesystem("file")
        self.fs_remote_fsspec = self._create_fs_remote(library="fsspec", warn=False)
        self.fs_remote_s3fs = self._create_fs_remote(library="s3fs")

    # ------------------------------------------------------------------------------------------------------------------
    def _create_fs_remote(self, library, warn=True):

        if library == "fsspec":
            if warn:
                warnings.warn("WARNING: Library `fsspec` cannot create asynchronous instances of file systems.")

            return fsspec.filesystem(
                protocol=self.base_fsspec_protocol,
                target_protocol=self.target_fsspec_protocol,
                target_options=dict(anon=True, endpoint_url=self.s3_endpoint_url),
                # cache_storage='.cache',  # Uncomment and define cache folder, if required.
                # asynchronous=True  # REMARK: This is allowed, but it does not make the instance asynchronous.
            )

        elif library == "s3fs":
            return s3fs.S3FileSystem(
                anon=True,
                endpoint_url=self.s3_endpoint_url,
                asynchronous=True
            )
        else:
            raise NotImplementedError(f"Error: Library {library} not supported.")

    # ------------------------------------------------------------------------------------------------------------------
    def list_all_shots(
            self,
            level: int = 2,
            test_data: bool = False,
            local: bool = False
    ):
        """
        Get a list of all shot indices.

        Parameters
        ----------
        level : int
            Target level for the MAST data to be pulled. Default: 2.
        test_data : bool
            If True, the target shot is pulled from test data, otherwise it is pulled from curated data. Not available
            for locally stored data (i.e, if local=True).
            Default: False.
        local : bool, optional
            If True, it checks locally stored data (e.g., in the CSD3 cluster), otherwise it looks into the registered
            remote data repository (e.g., a cloud S3 bucket).
            Default: False.

        Returns
        -------
            List of all available sources in the dataset.

        """

        if local:
            all_filenames = self.fs_local_fsspec.ls(f"{self.local_root_path}/level{level}/")
        else:
            
            if test_data:
                remote_path = f'/mast/test/level{level}/shots/'
            else:
                remote_path = f'/mast/level{level}/shots/'

            all_filenames = [item["Key"] for item in self.fs_remote_fsspec.ls(f"{remote_path}")]

        shot_ids = [
            filename.split("/")[-1].split(".zarr")[0] for filename in all_filenames if filename.endswith(".zarr")
        ]
        shot_ids = [
            int(shot_id) for shot_id in shot_ids if shot_id.isdigit()
        ]

        return shot_ids

    # ------------------------------------------------------------------------------------------------------------------
    def list_all_sources(
            self,
            shot_id: Union[NoneType, int] = None,
            level: int = 2,
            as_data_frame=False
    ):
        """
        List all sources via pd.read_parquet method.

        Parameters
        ----------
        shot_id : int
            ID of target shot to be checked in the MAST database. If not provided, all shots are checked.
            Default: None.
        level : int
            Target level for the MAST data to be pulled.
            Default: 2.
        as_data_frame : bool
            To return all sources as pandas data frame, otherwise only list of source names are returned.
            Default: False.

        Returns
        -------
            List of all available sources in the dataset.

        """

        assert level in [1, 2], "Input error: level must be in [1, 2]."
        if level == 1:
            raise NotImplementedError("Level 1 not yet supported.")

        if shot_id:
            assert isinstance(shot_id, int), "Type error: shot_id must be of type int."

        df = pd.read_parquet(path=f"{self.mast_app_url}/parquet/level{level}/sources")
        if shot_id:
            df = df.loc[df["shot_id"] == shot_id]

        if as_data_frame:
            return df
        else:
            return list(set(df["name"]))

    # ------------------------------------------------------------------------------------------------------------------
    def list_all_signals(
            self,
            shot_id: Union[NoneType, int] = None,
            level: int = 2,
            as_data_frame: bool = False
    ):
        """
        List all the available signals for a given target shot.

        Parameters
        ----------
        shot_id : int
            ID of target shot to be checked in the MAST database. If not provided, all shots are checked.
            Default: None.
        level : int
            Target level for the MAST data to be pulled.
            Default: 2.
        as_data_frame : bool
            To return shot signals as pandas data frame, otherwise only list of signal names are returned.
            Default: False.

        Returns
        -------
            List of all available signals for the target shot.

        REMARKS:
        --------
            - Time signals (e.g., time, time_bes, time_mirnov, time_omaha, time_saddle) are not retrieved as signals.
            - Current implementation relies on pd.read_parquet, meaning remote access is required.

        """

        assert level in [1, 2], "Input error: level must be in [1, 2]."
        if level == 1:
            raise NotImplementedError("Level 1 not yet supported.")

        if shot_id:
            assert isinstance(shot_id, int), "Type error: shot_id must be of type int."

        if shot_id:
            df = pd.read_parquet(f"{self.mast_app_url}/parquet/level{level}/signals?shot_id={shot_id}")
        else:
            shot_ids = self.list_all_shots(local=False)  # TODO: Double check for local.
            df = pd.read_parquet(f"{self.mast_app_url}/parquet/level{level}/signals?shot_id={shot_ids[0]}")
            for temp_shot_id in shot_ids[1:]:
                df = pd.concat([
                    df,
                    pd.read_parquet(f"{self.mast_app_url}/parquet/level{level}/signals?shot_id={temp_shot_id}")
                ])

        if as_data_frame:
            return df
        else:
            return list(df["name"])

    # ------------------------------------------------------------------------------------------------------------------
    def make_shot_store(
            self,
            shot_id: int,
            level: int = 2,
            test_data: bool = False,
            local: bool = False,
            verbose: bool = False
    ):
        """
        Get a store (either LocalStore or FsspecStore) for a given target shot.

        Parameters
        ----------
        shot_id : int
            ID of target shot to be pulled from MAST database.
        level : int
            Target level for the MAST data to be pulled.
            Default: 2.
        test_data : bool
            If True, the target shot is pulled from test data, otherwise it is pulled from curated data. Not available
            for locally stored data (i.e, if local=True).
            Default: False.
        local : bool
            If True, the target shot is pulled from locally stored data (e.g., in the CSD3 cluster), otherwise it is
            pulled from the registered remote data repository (e.g., a cloud S3 bucket).
            Default: False.
        verbose : bool
            If True, verbose mode is activated.
            Default: False.

        """

        assert level in [1, 2], "Input error: level must be in [1, 2]."

        if local:
            local_path = f"{self.local_root_path}/level{level}/{shot_id}.zarr"
            store = zarr.storage.LocalStore(root=local_path)
        else:
            remote_shot_path = f"{'test/' if test_data else ''}level{level}/shots/{shot_id}.zarr"
            store = zarr.storage.FsspecStore(
                fs=self.fs_remote_s3fs,
                read_only=True,
                path=f"/mast/{remote_shot_path}",
            )

        if verbose:
            print(f"{'LocalStore' if local else 'FsspecStore'} store for shot {shot_id} created.")

        return store

    # ------------------------------------------------------------------------------------------------------------------
    def make_shot_group(
            self,
            store: Union[ZarrStoreType, None] = None,
            shot_id: Union[int, None] = None,
            level: int = 2,
            local: bool = False,
            test_data: bool = False,
            verbose: bool = False
    ):
        """
        Create a shot group, either from a Zarr store, or directly from shot info.

        Parameters
        ----------
        store : ZarrStoreType or None
            Target Zarr store from which the group will be created. If not provided (i.e., store=None), then shot info
            is checked.
            Default: None.
        shot_id : int or None
            ID of target shot to be pulled from MAST database. Omitted if a valid store instance is provided.
            Default: None.
        level : int
            Target level for the MAST data to be pulled. Omitted if a valid store instance is provided.
            Default: 2.
        test_data : bool
            If True, the target shot is pulled from test data, otherwise it is pulled from curated data. Not available
            for locally stored data (i.e, if local=True). Omitted if a valid store instance is provided.
            Default: False.
        local : bool
            If True, the target shot is pulled from locally stored data (e.g., in the CSD3 cluster), otherwise it is
            pulled from the registered remote data repository (e.g., a cloud S3 bucket). Omitted if a valid store
            instance is provided.
            Default: False.
        verbose : bool
            If True, verbose mode is activated.
            Default: False.

        """

        if store:
            assert isinstance(store, ZarrStoreType), f"Invalid `store`: it must be of type `{ZarrStoreType}`."

            group = zarr.open_group(store=store, mode='r')

            if verbose:
                print(f"Group for store with path {store.path} created.")
        else:
            if shot_id is None:
                raise ValueError("Invalid `shot_id`: it must be a valid `int` value.")

            assert isinstance(shot_id, int), "Invalid `shot_id`: it must be of type `int`."

            if local:
                try:
                    self.fs_local_fsspec.ls(f"{self.local_root_path}/level{level}/")
                    # Create group by implicitly creating a writable LocalStore  # TODO: Double check for local.
                    local_path = f"{self.local_root_path}/level{level}/{shot_id}.zarr"
                except:
                    # Create group by implicitly creating a writable LocalStore  # TODO: Double check for local.
                    local_path = f"{self.local_root_path}/{shot_id}.zarr"
            
                group = zarr.create_group(store=local_path)
                # Source: https://zarr.readthedocs.io/en/latest/user-guide/storage.html#implicit-store-creation
                        

            else:
                # Create group by implicitly creating a read-only FsspecStore
                remote_shot_path = f"{'/test/' if test_data else ''}/level{level}/shots/{shot_id}.zarr"
                group = zarr.open_group(
                    store=f"{self.target_fsspec_protocol}://mast{remote_shot_path}",
                    mode="r",
                    storage_options={"anon": True, "endpoint_url": self.s3_endpoint_url}
                )
                # Source: https://zarr.readthedocs.io/en/latest/user-guide/storage.html#implicit-store-creation

            if verbose:
                print(f"Group for shot {shot_id} created.")

        if verbose:
            print("Group tree:\n")
            print(group.tree())

        return group


# ----------------------------------------------------------------------------------------------------------------------
def plot_1d_profiles(
        profiles: xr.Dataset
):
    """
    Helper function for plotting 1D profiles.

    """
    warnings.warn("WARNING: This method misbehaves if not run in a notebook cell with `%matplotlib notebook` header.")

    try:
        n = int(np.ceil(len(profiles.data_vars) / 2))
        fig, axes = plt.subplots(n, 2, figsize=(10, 2*n))
        axes = axes.flatten()

        for i, name in enumerate(profiles.data_vars.keys()):
            plot_1d_profile(profiles[name], ax=axes[i])

        plt.tight_layout()
        # plt.savefig(f"{xr.Dataset}.png")
        # plt.close()

    except Exception as e:
        print(f"Error: {e}")


# ----------------------------------------------------------------------------------------------------------------------
def plot_1d_profile(
        profile: xr.DataArray,
        ax=None
):
    try:

        profile.plot(x='time', ax=ax)

        if ax is None:
            ax = plt.gca()

        ax.grid('on', alpha=0.5)
        ax.set_xlim(profile.time.min(), profile.time.max())

        plt.show()

    except Exception as e:
        print(f"Error: {e}")


# ----------------------------------------------------------------------------------------------------------------------
def main():
    store_manager = MASTStorageManager()
    all_shots_ids = store_manager.list_all_shots(local=False)
    print(f"all_shots_ids: {all_shots_ids}\n")

    all_sources = store_manager.list_all_sources(shot_id=30471, as_data_frame=False)
    print(f"all_sources (as list): {all_sources}\n")

    all_sources = store_manager.list_all_sources(shot_id=30471, as_data_frame=True)
    print(f"all_sources (as data frame): {all_sources.to_string()}\n")
    print(f"all_sources.keys(): {all_sources.keys()}\n")

    all_signals = store_manager.list_all_signals(shot_id=30471, as_data_frame=False)
    print(f"all_signals (as list): {all_signals}\n")

    all_signals = store_manager.list_all_signals(shot_id=30471, as_data_frame=True)
    print(f"all_signals (as data frame): {all_signals.to_string()}\n")
    print(f"all_signals.keys(): {all_signals.keys()}\n")

    store_ = store_manager.make_shot_store(shot_id=30471, local=False)
    group_from_store = store_manager.make_shot_group(store=store_)
    print(f"group_from_store.tree() (group from store): {group_from_store.tree()}\n")

    group_from_shot_id = store_manager.make_shot_group(shot_id=30421, local=False)
    print(f"group_from_shot_id.tree() (group from shot if): {group_from_shot_id.tree()}\n")


# ======================================================================================================================
if __name__ == "__main__":
    main()
