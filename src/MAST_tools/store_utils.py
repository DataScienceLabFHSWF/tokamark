"""
Docstring reference: https://numpydoc.readthedocs.io/en/latest/format.html
Python style reference: https://google.github.io/styleguide/pyguide.html
"""

import os
import numpy as np
import zarr
import zarr.storage
import fsspec
import s3fs
import pandas as pd
import warnings
from typing import Union, Optional, Any
import logging
from urllib.error import HTTPError
from pprint import pprint
import time

try:
    from . import constants as cc
except ImportError:
    import constants as cc

logging.getLogger('asyncio').setLevel(logging.CRITICAL)


# ======================================================================================================================
class MASTStorageManager:
    """
    Class with storage management tools for MAST data.

    Attributes
    ----------
    base_fsspec_protocol : str
        Base protocol used by fsspec.
    target_fsspec_protocol : str
        Target filesystem protocol for the selected base fsspec protocol.
    s3_endpoint_url : str
        Endpoint of the cloud S3 bucket used for remote data pulling.
    base_remote_parquet_path : Optional[str]
        Local root path used for remote metadata pulling in parquet format.
    base_local_zarr_path : Optional[str]
        Local root path used for local data pulling in zarr format.
    base_local_parquet_path : Optional[str]
        Local root path used for local metadata pulling in parquet format.
    fs_local_fsspec : fsspec.implementations.local.LocalFileSystem
        A LocalFileSystem instance.
    fs_remote_fsspec : Any
        A filesystem instance via fsspec.filesystem.
    fs_remote_s3fs : Any
        A filesystem instance via s3fs.S3FileSystem.
    store_manager_id : str
        User-defined storage manager ID.

    Methods
    -------
    _is_digit(item)
        Check if provided item is of type digit.
    _get_store_from_data_origin(data_origin)
        Auxiliary function to get Zarr store instance from a given data origin.
    _parse_shot_info_dict(shot_info)
        Parse dictionary with shot information.
    _check_level(level)
        Check MAST data level.
    _check_data_origin(data_origin)
        Check MAST data origin.
    _check_shot_id(shot_id)
        Check shot ID.
    _check_list_of_shot_ids(shot_ids)
        Check list of shot IDs.
    _create_fs_remote(library, warn)
        Create filesystem instance either using fsspec or s3fs for remote data request.
    _read_parquet_data(path, local)
        Read MAST data using parquet pipeline.
    _read_fsspec_listdir(path, local)
        Evaluate the list dir method of fsspec filesystem instance on the provided path.
    list_all_shots(level, test_data, local, via_parquet)
        Get a list of available MAST shot IDs.
    list_shots_by_signal_availability(availability_data_filepath, required_signals)
        List shot IDs following composite condition for signal availability and given availability file.
    get_all_sources(shot_ids, level, test_data, local, via_parquet)
        Return a dictionary with all available sources per shot ID.
    get_all_signals(shot_ids, level, test_data, local, via_parquet)
        Return a dictionary with all available signals per shot ID.
    make_shot_store(shot_id, verbose)
        Make a Zarr store (either LocalStore or FsspecStore) for a given target shot.
    make_shot_group(data_origin, verbose)
        Make a shot group from data origin (either a Zarr store or shot info).

    # Pending methods  # TODO: Check if needed.
    # ---------------
    #
    # print_store(...)
    # is_signal_in_store(...)
    # print_signals_in_group(...)

    """

    # ------------------------------------------------------------------------------------------------------------------
    def __init__(
            self,
            base_fsspec_protocol: str = "simplecache",
            target_fsspec_protocol: str = "s3",
            s3_endpoint_url: str = "https://s3.echo.stfc.ac.uk",
            base_remote_parquet_path: Optional[str] = "https://mastapp.site/parquet",
            base_local_zarr_path: Optional[str] = "/rds/project/rds-mOlK9qn0PlQ/fairmast/upload-tmp",
            base_local_parquet_path: Optional[str] = "../../metadata/parquet",
            store_manager_id: str = ""
    ) -> None:
        """
        Initialise class attributes.

        Parameters
        ----------
        base_fsspec_protocol: str
            Base protocol used by fsspec. Some supported protocols include:
            `blockcache`:
                - With this option, data is downloaded block-wise.
                - Restrictions:
                    - It has a storage/OS combination which supports sparse files.
                    - The backend implementation uses files which derive from AbstractBufferedFile.
                    - The library you pass the resultant object to accepts generic python file-like objects.
            `filecache`:
                - Works for all file system implementations, and provides a real local file for other libraries to use.
            `simplecache`:
                - Same as `filecache`, except without options for cache expiry and to check original source.
                - Only option guaranteed to be thread/process-safe.
            Full list of supported protocols obtained via `fsspec.available_protocols()`.
            More info available at https://filesystem-spec.readthedocs.io/en/latest/features.html.
            Default: "simplecache".
        target_fsspec_protocol : str
            Target filesystem protocol for the selected base fsspec protocol.
            Default: "s3".
        s3_endpoint_url : str
            Endpoint of the cloud S3 bucket used for remote data pulling (i.e., for local=False in some methods).
            Default: "https://s3.echo.stfc.ac.uk".
        base_remote_parquet_path : Optional[str]
            Local root path used for remote metadata pulling (i.e., for local=False in some methods) in parquet format.
            Default: "https://mastapp.site/parquet/".
        base_local_zarr_path : Optional[str]
            Local root path used for local data pulling in zarr format.
            Default: "/rds/project/rds-mOlK9qn0PlQ/fairmast/upload-tmp", corresponding to the CSD3 cluster.
            FIXME: Should this default value be included when open sourcing?
        base_local_parquet_path : Optional[str]
            Local root path used for local metadata pulling (i.e., for local=True in some methods) in parquet format.
            Default: None.
        store_manager_id : str
            User defined manager ID. Default: "".

        Returns
        -------
        None

        """

        self.base_fsspec_protocol = base_fsspec_protocol
        self.target_fsspec_protocol = target_fsspec_protocol
        self.s3_endpoint_url = s3_endpoint_url
        self.base_remote_parquet_path = base_remote_parquet_path
        self.base_local_zarr_path = base_local_zarr_path
        self.base_local_parquet_path = base_local_parquet_path

        self.fs_local_fsspec = fsspec.filesystem("file")
        self.fs_remote_fsspec = self._create_fs_remote(library="fsspec")
        self.fs_remote_s3fs = self._create_fs_remote(library="s3fs")

        self.store_manager_id = store_manager_id

    # ------------------------------------------------------------------------------------------------------------------
    def build_level_path(self, level: int | str, test_data: bool) -> str:
        level_str = f"level{level}"
        return os.path.join("test", level_str) if test_data else level_str

    @staticmethod
    def _is_digit(
            item: Any
    ) -> bool:
        """
        Check if provided item is of type digit.

        Parameters
        ----------
        item : Any
            Any arbitrary item.

        Returns
        -------
        bool
            True if `item` is digit (either of int type, or of type str representing a digit), False otherwise.

        """

        is_digit = False
        if isinstance(item, str):
            if item.isdigit():
                is_digit = True
        elif isinstance(item, (int, np.int_)):
            is_digit = True

        return is_digit

    # ------------------------------------------------------------------------------------------------------------------
    def _get_store_from_data_origin(
            self,
            data_origin: Union[dict, cc.ZarrStoreType],
    ) -> zarr.storage.FsspecStore:
        """
        Auxiliary function to get Zarr store instance from a given data origin.

        Parameters
        ----------
        data_origin : Union[dict, cc.ZarrStoreType]
            Origin of data for group creation. It can be a dictionary with shot information (as in the class method
            `self.make_shot_store()`) or a Zarr store.

        Returns
        -------
        zarr.storage.FsspecStore
            Instance of Zarr store.

        Raises
        ------
        None

        """

        self._check_data_origin(data_origin=data_origin)
        if isinstance(data_origin, dict):
            # data_origin is a dict with shot info
            return self.make_shot_store(shot_info=data_origin)
        else:
            # data_origin is a Zarr store
            return data_origin

    # ------------------------------------------------------------------------------------------------------------------
    def _parse_shot_info_dict(
            self,
            shot_info: dict,
            enforce_shot_id: bool = True
    ) -> dict:
        """
        Parse dictionary with shot information.

        Parameters
        ----------
        shot_info : dict
            Dictionary with shot information required for store creation. All dictionary keys and types are as follows:
                "shot_id : int
                    ID of target shot to be pulled from MAST database.
                "level" : int
                    Target level for the MAST data to be pulled.
                    Optional. Default: 2.
                "test_data" : bool
                    If True, the target shot is pulled from test data, otherwise it is pulled from curated data. Not
                    available for locally stored data (i.e, if local=True).
                    Optional. Default: False.
                "local" : bool
                    If True, the target shot is pulled from locally stored data (e.g., in the CSD3 cluster), otherwise
                    it is pulled from the registered remote data repository (e.g., a cloud S3 bucket).
                    Optional. Default: False.
        enforce_shot_id : bool
            Boolean flag to enforce checking of 'shot_id' field in `shot_info`. If False, 'shot_id' is not checked.
            Optional. Default: True.

        Returns
        -------
        dict
            Dictionary with parsed items.

        Raises
        ------
        TypeError
            If parameter `shot_info` is not a dict, or if fields in `shot_info` ['local', 'test_data'] are not boolean.

        """

        parsed_shot_info = {"level": 2, "local": False, "test_data": False}

        if "shot_id" in shot_info:
            parsed_shot_info["shot_id"] = shot_info.get("shot_id")
            self._check_shot_id(shot_id=parsed_shot_info["shot_id"])
        else:
            if enforce_shot_id:
                raise KeyError("Missing field 'shot_id'.")

        if "level" in shot_info:
            parsed_shot_info["level"] = shot_info.get("level")
            self._check_level(level=parsed_shot_info["level"])

        if "local" in shot_info:
            parsed_shot_info["local"] = shot_info.get("local")
            if not isinstance(parsed_shot_info["local"], bool):
                raise TypeError("Invalid field 'local': it must be of type bool.")

        if "test_data" in shot_info:
            parsed_shot_info["test_data"] = shot_info.get("test_data")
            if not isinstance(parsed_shot_info["test_data"], bool):
                raise TypeError("Invalid field 'test_data': it must be of type bool.")

        return parsed_shot_info

    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def _check_level(
            level: int,
    ) -> None:
        """
        Check MAST data level.

        Parameters
        ----------
        level : int
            Target MAST data level to be checked.

        Returns
        -------
        None

        Raises
        ------
        None

        """

        if level not in [1, 2]:
            raise ValueError("Invalid parameter 'level': it must be in [1, 2].")

    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def _check_data_origin(
            data_origin: Union[dict, cc.ZarrStoreType],
    ) -> None:
        """
        Check MAST data origin.

        Parameters
        ----------
        data_origin : Union[dict, cc.ZarrStoreType]
            Object expected to define data origin for Zarr group/store creation.

        Returns
        -------
        None

        Raises
        ------
        TypeError
            If parameter data_origin is not dict or `ZarrStoreType` (i.e., either `zarr.storage.FsspecStore` or
            `zarr.storage.LocalStore`).

        """

        if not isinstance(data_origin, cc.DataSourceType):
            raise TypeError("Invalid parameter 'data_origin': it must be dict or store.")

    # ------------------------------------------------------------------------------------------------------------------
    def _check_shot_id(
            self,
            shot_id: Any
    ) -> None:
        """
        Check if provided shot ID has a valid type/value. Allowed types are int, or str with value being a digit.

        Parameters
        ----------
        shot_id : Any
            Target shot ID.

        Returns
        -------
        None

        Raises
        ------
        TypeError
            If parameter `shot_id` not of type int or str.
        ValueError
            If parameter `shot_id` is of type str and its value is not a digit.

        """

        if not self._is_digit(item=shot_id):
            raise ValueError("Invalid field/parameter 'shot_id': it must be of type int, or of type str representing a "
                             "digit.")

    # ------------------------------------------------------------------------------------------------------------------
    def _check_list_of_shot_ids(
            self,
            shot_ids: Union[list[int], tuple[int]],
    ) -> None:
        """
        Check list of shot IDs.

        Parameters
        ----------
        shot_ids : Union[list[int], tuple[int]]
            List of shot IDs.

        Returns
        -------
        None

        Raises
        ------
        None

        """

        for id_ in shot_ids:
            self._check_shot_id(shot_id=id_)

    # ------------------------------------------------------------------------------------------------------------------
    def _create_fs_remote(
            self,
            library: str,
            warn: bool = False
    ) -> Any:
        """
        Create a filesystem instance either using 'fsspec' or 's3fs' libraries for remote data request.

        Parameters
        ----------
        library : str
            The target library for filesystem instance creation.
        warn : bool
            If True, warn about issues during filesystem instance creation.
            Default: False.

        Returns
        -------
        Any
            A filesystem instance, created via either `fsspec.filesystem` or `s3fs.S3FileSystem`.

        Raises
        ------
        NotImplementedError
            If a library other than 'fsspec' or 's3fs' is provided.

        """

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
    @staticmethod
    def _read_parquet_data(
            path: str,
            local: bool
    ) -> pd.DataFrame:
        """
        Read MAST data from a target file path using parquet pipeline.

        Parameters
        ----------
        path : str
            Target file path.
        local: bool
            Boolean flag to activate local mode. If True, use local data, otherwise use remote data.

        Returns
        -------
        pd.DataFrame
            Pandas dataframe with the target parquet data.

        Raises
        ------
        FileNotFoundError
           If no parquet data is available for the provided path.

        """

        local_tag = "local" if local else "remote"

        try:
            return pd.read_parquet(path=path)
        except HTTPError as ee:
            raise FileNotFoundError(f"No data available for {local_tag} path {path} ({ee}).")
        except FileNotFoundError as ee:
            raise FileNotFoundError(f"No data available for {local_tag} path {path} ({ee}).")

    # ------------------------------------------------------------------------------------------------------------------
    def _read_fsspec_listdir(
            self,
            path: str,
            local: bool = False
    ) -> list:
        """
        Evaluate the list dir method of fsspec filesystem instance on the provided path.

        Parameters
        ----------
        path : str
            Target path to be evaluated via the invoked list dir method.
        local : str
            Boolean flag to define the fsspec instance to be used. If True, it corresponds to `self.fs_local_fsspec`,
            i.e., a `fsspec.implementations.local.LocalFileSystem` instance; otherwise, `self.fs_remote_fsspec` is used,
            which corresponds to a fsspec instance created either via `fsspec.filesystem` or via `s3fs.S3FileSystem`.
            Default: False.

        Returns
        -------
        list
            List of items in filesystem instance.

        Raises
        ------
        FileNotFoundError
            If invalid path is provided.

        """
        try:
            if local:
                return self.fs_local_fsspec.ls(path)
            else:
                return self.fs_remote_fsspec.ls(path)
        except FileNotFoundError:
            raise FileNotFoundError(f"No data available for path {path}.")

    # ------------------------------------------------------------------------------------------------------------------
    def list_all_shots(
            self,
            level: int = 2,
            test_data: bool = False,
            local: bool = False,
            via_parquet: bool = False
    ) -> list:
        """
        Get a list of available MAST shot IDs.

        Parameters
        ----------
        level : int
            Target level for the MAST data to be pulled. Default: 2.
        test_data : bool
            If True, the target shot is pulled from test data, otherwise it is pulled from curated data.
            Optional. Default: False.
        local : bool
            If True, it checks locally stored data (e.g., in the CSD3 cluster), otherwise it looks into the registered
            remote data repository (e.g., a cloud S3 bucket).
            Optional. Default: False.
        via_parquet : bool
            If True, it retrieves information via `pandas.read_parquet` pipeline. Otherwise, alternative pipeline (e.g.,
            `fsspec.filesystem`) is used.
            Optional: Default: False.

        Returns
        -------
        list
            List of all available sources in the dataset.

        Raises
        ------
        ValueError
            If `via_parquet` is True and parquet path is None.

        """

        self._check_level(level=level)
        test_and_level_case = self.build_level_path(level, test_data)

        if via_parquet:
            # Parquet pipeline
            base_parquet_path = self.base_local_parquet_path if local else self.base_remote_parquet_path
            if base_parquet_path is None:
                raise ValueError(f"Invalid value '{base_parquet_path}' for attribute 'base_local_parquet_path' of the "
                                 f"MASTStorageManager instance.")

            if local:
                full_path = os.path.join(base_parquet_path,test_and_level_case,"shots_metadata.parquet")
            else:
                full_path = os.path.join(base_parquet_path,test_and_level_case,"shots")

            summary_dataframe = self._read_parquet_data(path=full_path, local=local)
            raw_shot_ids = summary_dataframe["shot_id"].values
        else:
            # FSSpec pipeline
            if local:
                local_path = os.path.join(self.base_local_zarr_path,test_and_level_case)
                all_filenames = self._read_fsspec_listdir(path=local_path, local=True)
            else:
                remote_path = os.path.join("mast",test_and_level_case,"shots")
                all_filenames = [item["Key"] for item in self._read_fsspec_listdir(path=remote_path, local=False)]

            raw_shot_ids = [
                filename.split("/")[-1].split(".zarr")[0] for filename in all_filenames if filename.endswith(".zarr")
            ]

        shot_ids = [
            int(raw_shot_id) for raw_shot_id in raw_shot_ids if self._is_digit(raw_shot_id)
        ]
        shot_ids.sort()

        return shot_ids

    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def list_shots_by_signal_availability(
            availability_data_filepath: str,
            required_signals: dict
    ) -> list:
        """
        List shot IDs following composite condition for signal availability and given availability file.

        Parameters
        ----------
        availability_data_filepath : str
            Path to suitable csv file with signal availability.
        required_signals: dict
            Dictionary with required signal availability.
            Example: {"thomson_scattering": ["n_e"], "summary": ["power_nbi", "ip"]}

        Returns
        -------
        list
            List of shot IDs.

        Raises
        ------
        FileNotFoundError
            If invalid availability_data_filepath is provided.

        """

        try:
            availability_data = pd.read_csv(availability_data_filepath)
        except FileNotFoundError:
            raise FileNotFoundError(f"Invalid availability_data_filepath '{availability_data_filepath}'.")

        source_signals_to_have = []
        for kk, vv in required_signals.items():
            source_signals_to_have += ([f"{kk}__{val}" for val in vv])

        # REMARK: Suggested "is True" for line below causes error due to misbehaviour, thus noqa in place.
        composite_condition = availability_data[source_signals_to_have[0]] == True  # noqa

        for signal in source_signals_to_have[1:]:
            # REMARK: Suggested "is True" for line below causes error due to misbehaviour, thus noqa in place.
            composite_condition = composite_condition & (availability_data[signal] == True)  # noqa

        return list(availability_data.loc[composite_condition]["shot_id"])

    # ------------------------------------------------------------------------------------------------------------------
    def get_all_sources(
            self,
            shot_ids: Optional[list[int]] = None,
            level: int = 2,
            test_data: bool = False,
            local: bool = False,
            via_parquet: bool = False
    ) -> dict:
        """
        Return a dictionary with all available sources per shot ID.

        Parameters
        ----------
        shot_ids : Optional[list[int]]
            Target shot IDs to be checked in the MAST database. If None is provided, all available shots are checked.
            Optional. Default: None.
        level : int
            Target level for the MAST data to be pulled.
            Optional. Default: 2.
        test_data : bool
            If True, the target shot is pulled from test data, otherwise it is pulled from curated data.
            Optional. Default: False.
        local : bool
            If True, it checks locally stored data (e.g., in the CSD3 cluster), otherwise it looks into the registered
            remote data repository (e.g., a cloud S3 bucket).
            Optional. Default: False.
        via_parquet : bool
            If True, it retrieves information via `pandas.read_parquet` pipeline. Otherwise, alternative pipeline (e.g.,
            `fsspec.filesystem`) is used.
            Optional. Default: False.

        Returns
        -------
        dict
            Dictionary with available sources per shot ID.

        Raises
        ------
        ValueError
            If `via_parquet` is True and parquet path is None.

        """

        self._check_level(level=level)
        if shot_ids is not None:
            self._check_list_of_shot_ids(shot_ids=shot_ids)

        test_and_level_case = self.build_level_path(level, test_data)

        if via_parquet:
            # Parquet pipeline

            base_parquet_path = self.base_local_parquet_path if local else self.base_remote_parquet_path
            if base_parquet_path is None:
                raise ValueError(f"Invalid value '{base_parquet_path}' for attribute 'base_local_parquet_path' of the "
                                 f"MASTStorageManager instance.")

            if local:
                full_path = os.path.join(base_parquet_path,test_and_level_case,"shots_sources.parquet")
            else:
                full_path = os.path.join(base_parquet_path,test_and_level_case,"sources")
            summary_dataframe = self._read_parquet_data(path=full_path, local=local)
            if shot_ids:
                composite_condition = summary_dataframe["shot_id"] == shot_ids[0]
                for shot_id in shot_ids[1:]:
                    composite_condition = composite_condition | (summary_dataframe["shot_id"] == shot_id)
                summary_dataframe = summary_dataframe.loc[composite_condition]

            source_info = {}
            all_shot_ids = list(set(summary_dataframe["shot_id"]))
            for id_ in all_shot_ids:
                sub_dataframe = summary_dataframe.loc[summary_dataframe["shot_id"] == id_]
                source_info[id_] = list(sub_dataframe["name"])

            return source_info

        else:
            # FSSpec pipeline

            if shot_ids is None:
                shot_ids = self.list_all_shots(level=level, test_data=test_data, local=local, via_parquet=via_parquet)

            source_info = {}
            for id_ in shot_ids:
                group = self.make_shot_group(
                    data_origin={
                        "shot_id": id_,
                        "level": level,
                        "test_data": test_data,
                        "local": local
                    }
                )
                source_info[id_] = list(group.keys())

            return source_info

    # ------------------------------------------------------------------------------------------------------------------
    def get_all_signals(
            self,
            shot_ids: Optional[list[int]] = None,
            level: int = 2,
            test_data: bool = False,
            local: bool = False,
            via_parquet: bool = False,
            verbose: bool = False
    ) -> dict:
        """
        Return a dictionary with all available signals per shot ID.

        Parameters
        ----------
        shot_ids : Optional[list of ints]
            Target shot IDs to be checked in the MAST database. If None is provided, all shots are checked.
            Default: None.
        level : int
            Target level for the MAST data to be pulled.
            Default: 2.
        test_data : bool
            If True, the target shot is pulled from test data, otherwise it is pulled from curated data.
            Default: False.
        local : bool
            If True, it checks locally stored data (e.g., in the CSD3 cluster), otherwise it looks into the registered
            remote data repository (e.g., a cloud S3 bucket).
            Optional. Default: False.
        via_parquet : bool
            If True, it retrieves information via pandas.read_parquet pipeline. Otherwise, alternative pipeline (e.g.,
            fsspec.filesystem) is used.
            Optional. Default: False.
        verbose : bool
            If True, verbose mode is activated.
            Default: False.

        Returns
        -------
        dict
            Dictionary with available signals per shot ID.

        Raises
        ------
        ValueError
            If 'via_parquet' is True and parquet path is None.

        Notes
        -----
        - Time signals (e.g., time, time_bes, time_mirnov, time_omaha, time_saddle) are not retrieved as signals when
          `via_parquet` is True.

        """

        self._check_level(level=level)
        if shot_ids is None:
            shot_ids = self.list_all_shots(level=level, test_data=test_data, local=local, via_parquet=via_parquet)
        else:
            self._check_list_of_shot_ids(shot_ids=shot_ids)

        test_and_level_case = self.build_level_path(level, test_data)
        
        if via_parquet:
            # Parquet pipeline

            base_parquet_path = self.base_local_parquet_path if local else self.base_remote_parquet_path
            if base_parquet_path is None:
                raise ValueError(f"Invalid value '{base_parquet_path}' for attribute 'base_local_parquet_path' of the "
                                 f"MASTStorageManager instance.")

            signal_info = {}
            if local:
                root_signals_path = os.path.join(base_parquet_path,test_and_level_case,"all_signals")

                all_signal_names = os.listdir(root_signals_path)
                len_all_signals = len(all_signal_names)

                for ii_signal, signal_name in enumerate(all_signal_names):
                    signal_path = f"{root_signals_path}/{signal_name}"
                    signal_df = self._read_parquet_data(path=signal_path, local=local)
                    data_tuples = [
                        (row['shot_id'], f"{row['source']}__{row['name']}") for index, row in signal_df.iterrows()
                        if row["shot_id"] in shot_ids
                    ]

                    for data_tuple in data_tuples:
                        shot_id_, source_signal_ = data_tuple
                        try:
                            if source_signal_ not in signal_info[shot_id_]:
                                signal_info[shot_id_].append(source_signal_)
                        except KeyError:
                            signal_info[shot_id_] = [source_signal_]

                    if verbose:
                        print(f"Processed signals: {ii_signal} out of {len_all_signals}.")

            else:
                len_shot_ids = len(shot_ids)
                for ii, shot_id_ii in enumerate(shot_ids):
                    full_path = f"{base_parquet_path}/{test_and_level_case}/signals?shot_id={shot_id_ii}"
                    df = self._read_parquet_data(path=full_path, local=local)
                    signal_info[shot_id_ii] = [f"{row['source']}__{row['name']}" for index, row in df.iterrows()]

                    if verbose:
                        print(f"Processed shots: {ii} out of {len_shot_ids}.")

            return signal_info

        else:
            # FSSpec pipeline

            signal_info = {}
            for shot_id in shot_ids:

                group = self.make_shot_group(
                    data_origin={
                        "shot_id": shot_id,
                        "level": level,
                        "test_data": test_data,
                        "local": local,
                    }
                )

                group_members = group.members(1)
                for item in group_members:
                    member = None
                    try:
                        group, member = item[0].split("/")
                        signal_info[shot_id].append(f"{group}__{member}")
                    except KeyError:
                        signal_info[shot_id] = [f"{group}__{member}"]
                    except ValueError:
                        pass
                        # print(f"Skipped item: {item[0]}")

            return signal_info

    # ------------------------------------------------------------------------------------------------------------------
    def make_shot_store(
            self,
            shot_info: dict,
            verbose: bool = False
    ) -> Union[zarr.storage.LocalStore, zarr.storage.FsspecStore]:
        """
        Make a Zarr store (either LocalStore or FsspecStore) for a given target shot.

        Parameters
        ----------
        shot_info : dict
            Dictionary with shot information required for store creation. All dictionary keys and types are as follows:

                "shot_id" : int
                    ID of target shot to be pulled from MAST database.
                "level" : int
                    Target level for the MAST data/metadata to be pulled.
                    Optional. Default: 2.
                "test_data" : bool
                    If True, the target shot is pulled from test data, otherwise it is pulled from curated data. Not
                    available for locally stored data (i.e, if `local` is True).
                    Optional. Default: False.
                "local" : bool
                    If True, the target shot is pulled from locally stored data (e.g., in the CSD3 cluster), otherwise
                    it is pulled from the registered remote data repository (e.g., a cloud S3 bucket).
                    Optional. Default: False.

        verbose : bool
            If True, verbose mode is activated.
            Default: False.

        Returns
        -------
        Union[zarr.storage.LocalStore, zarr.storage.FsspecStore]
            Either a zarr.storage.LocalStore instance or a zarr.storage.FsspecStore instance.

        Raises
        ------
        None

        """

        parsed_shot_info = self._parse_shot_info_dict(shot_info=shot_info)
        test_and_level_case = self.build_level_path(parsed_shot_info["level"],parsed_shot_info["test_data"])

        if parsed_shot_info["local"]:
            zarr_filepath = os.path.join(self.base_local_zarr_path,test_and_level_case,f"{parsed_shot_info['shot_id']}.zarr")
            
            store = zarr.storage.LocalStore(root=zarr_filepath)
        else:
            zarr_filepath = os.path.join( "mast",test_and_level_case,"shots",f"{parsed_shot_info['shot_id']}.zarr" )
            
            store = zarr.storage.FsspecStore(
                fs=self.fs_remote_s3fs,
                read_only=True,
                path=zarr_filepath
            )

        # if verbose:
        #     store_type = 'LocalStore' if parsed_shot_info["local"] else 'FsspecStore'
        #     print(f"{store_type} store for shot {parsed_shot_info['shot_id']} created.")

        return store

    # ------------------------------------------------------------------------------------------------------------------
    def make_shot_group(
            self,
            data_origin: Union[dict, cc.ZarrStoreType],
            verbose: bool = False
    ) -> zarr.Group:
        """
        Make a shot group from data origin (either a Zarr store or shot info).

        Parameters
        ----------
        data_origin : Union[dict, ZarrStoreType]
            Origin of data for group creation. It can be a dictionary with shot information (as in the class method
            `self.make_shot_store()`) or a Zarr store.
        verbose : bool
            If True, verbose mode is activated.
            Default: False.

        Returns
        -------
        zarr.Group
            Zarr group created from data origin.

        Raises
        ------
        None

        """

        self._check_data_origin(data_origin=data_origin)
        if isinstance(data_origin, cc.ZarrStoreType):
            # Create group from store
            store = data_origin
            group = zarr.open_group(store=store, mode='r')
            if verbose:
                print(f"Group for store with path {store.path} created.")
        else:
            # Create group from shot info dictionary.

            parsed_shot_info = self._parse_shot_info_dict(shot_info=data_origin)
            test_and_level_case = self.build_level_path(parsed_shot_info["level"],parsed_shot_info["test_data"])
            
            if parsed_shot_info['local']:
                # Create group by implicitly creating a writable LocalStore
                local_path = os.path.join( self.base_local_zarr_path, test_and_level_case,f"{parsed_shot_info['shot_id']}.zarr")
                    
                group = zarr.open_group(store=local_path, mode="r")
                # Source: https://zarr.readthedocs.io/en/latest/user-guide/storage.html#implicit-store-creation
            else:
                # Create group by implicitly creating a read-only FsspecStore
                
                remote_shot_path = os.path.join("/mast",test_and_level_case, "shots", f"{parsed_shot_info['shot_id']}.zarr")

                group = zarr.open_group(
                    store=f"{self.target_fsspec_protocol}:/{remote_shot_path}",
                    mode="r",
                    storage_options={"anon": True, "endpoint_url": self.s3_endpoint_url}
                )
                # Source: https://zarr.readthedocs.io/en/latest/user-guide/storage.html#implicit-store-creation

            if verbose:
                print(f"Group for shot {parsed_shot_info['shot_id']} created.")

        if verbose:
            print("Group tree:\n")
            print(group.tree())

        return group


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

    # Base settings
    shot_ids = [30471]  # None for entire dataset (which takes very long time if local=False).
    shot_id = 30471
    level = 2
    test_data = False
    local = True
    via_parquet = True
    verbose = False

    # Creation of MASTStorageManager instance for test
    store_manager = MASTStorageManager(
        base_fsspec_protocol="simplecache",
        target_fsspec_protocol="s3",
        s3_endpoint_url="https://s3.echo.stfc.ac.uk",
        base_remote_parquet_path="https://mastapp.site/parquet",
        base_local_zarr_path="/rds/project/rds-mOlK9qn0PlQ/fairmast/upload-tmp",
        base_local_parquet_path="../../metadata/parquet",
        store_manager_id="my_store_manager"
    )

    TESTS_TO_RUN = {  # noqa
        "get_all_shot_ids": False,
        "get_all_sources": False,
        "get_all_signals": False,
        "make_group_from_store": False,
        "make_group_from_shot_info": False,
        "check_signal_availability": True
    }

    # ..................................................................................................................
    # List all shot IDs for the entire dataset, using different pipelines

    if TESTS_TO_RUN["get_all_shot_ids"]:

        pipeline_tag = "parquet" if via_parquet else "ffspec"
        local_tag = "local" if local else "remote"
        test_tag = ", test" if test_data else ""
        print(f"Getting available shot IDs ({pipeline_tag} pipeline, {local_tag} bucket, level {level}{test_tag})...\n")
        t0 = time.time()
        all_shots_ids = store_manager.list_all_shots(
            level=level,
            test_data=test_data,
            local=local,
            via_parquet=via_parquet
        )

        print(f"Number of shots: {len(all_shots_ids)}")
        print(f"Shot IDs: {list(all_shots_ids)}")
        print(f"Elapsed time: {round(time.time() - t0, 2)} s\n")

    # ..................................................................................................................
    # List all sources

    if TESTS_TO_RUN["get_all_sources"]:

        all_sources = store_manager.get_all_sources(
            shot_ids=shot_ids,
            level=level,
            test_data=test_data,
            local=local,
            via_parquet=via_parquet
        )
        print(f"len(all_sources): {len(all_sources)}\n")
        # pprint(all_sources)

    # ..................................................................................................................
    # List all signals

    if TESTS_TO_RUN["get_all_signals"]:

        all_signals = store_manager.get_all_signals(
            shot_ids=shot_ids,
            level=level,
            test_data=test_data,
            local=local,
            via_parquet=via_parquet,
            verbose=verbose
        )
        print(f"len(all_signals): {len(all_signals)}")
        # pprint(all_signals)

    # ..................................................................................................................
    # Make group for a given shot

    # Via existing store object
    if TESTS_TO_RUN["make_group_from_store"]:

        store_ = store_manager.make_shot_store(
            shot_info={
                "shot_id": shot_id,
                "level": level,
                "test_data": test_data,
                "local": local
            }
        )
        group_from_store = store_manager.make_shot_group(data_origin=store_)
        print(f"group_from_store.tree() (group from store): {group_from_store.tree()}\n")

    # Directly from shot_info
    if TESTS_TO_RUN["make_group_from_shot_info"]:

        group_from_shot_id = store_manager.make_shot_group(
            data_origin={"shot_id": 30471, "level": 2, "test_data": False, "local": False}
        )
        print(f"group_from_shot_id.tree() (group from shot ID): {group_from_shot_id.tree()}\n")

    # ..................................................................................................................
    # List all shots IDs for given signal availability

    if TESTS_TO_RUN["check_signal_availability"]:

        signal_availability_file = "../../metadata/2025-04-17/data_level2_signal_availability.csv"
        dict_target_signals = {
            "thomson_scattering": ["n_e"],
            "spectrometer_visible": ["filter_spectrometer_bes_voltage"],
            "summary": ["power_nbi", "ip"],
        }

        print("\nSignals to be checked for simultaneous availability across all shots:")
        pprint(dict_target_signals)

        filtered_ids = store_manager.list_shots_by_signal_availability(
            availability_data_filepath=signal_availability_file,
            required_signals=dict_target_signals
        )

        print(f"\nfiltered_ids ({len(filtered_ids)} shots):")
        pprint(filtered_ids)

    # ..................................................................................................................

    print("---------------------------------------------")
    print(f"Elapsed time for tests() execution: {round(time.time() - t0_all_tests, 2)} s")


# ======================================================================================================================
if __name__ == "__main__":
    tests()
