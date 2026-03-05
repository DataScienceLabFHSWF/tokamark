"""
Docstring reference: https://numpydoc.readthedocs.io/en/latest/format.html
Python style reference: https://google.github.io/styleguide/pyguide.html
"""

from typing import Union, Any
from collections.abc import Mapping
from typing_extensions import Unpack
import zarr.storage
from xarray.core.dataset import Dataset
from pathlib import Path
from os.path import join as os_join

from MAST_tools.data_models import StoreManagerParameters, ShotInfo

# ======================================================================================================================
# Data types

ShotInfoType = Union[Mapping[str, Any], Unpack[ShotInfo]]
StoreManagerParametersType = Union[Mapping[str, Any], Unpack[StoreManagerParameters]]

XarrayDatasetType = Dataset
ZarrFSStoreType = zarr.storage.FsspecStore
ZarrLocalStoreType = zarr.storage.LocalStore
ZarrStoreType = Union[ZarrFSStoreType, ZarrLocalStoreType]
BaseDataSourceType = Union[ShotInfoType, ZarrStoreType]
ExtendedDataSourceType = Union[BaseDataSourceType, XarrayDatasetType]

# ======================================================================================================================
# Default directories

PROJECT_ROOT_DIR = Path(__file__).parent.parent.parent
METADATA_DIR = os_join(PROJECT_ROOT_DIR, "metadata")
SCRIPTS_DIR = os_join(PROJECT_ROOT_DIR, "scripts")

# ======================================================================================================================
# Default file paths

DEFAULT_SIGNAL_AVAILABILITY_FILE = os_join(METADATA_DIR, "sources_signals", "signal_availability.csv")
DEFAULT_SOURCES_WITH_SIGNALS_FILE = os_join(METADATA_DIR, "sources_signals", "dict_sources_with_signals.yaml")

DEFAULT_SIGNALS_STATS_FILE = os_join(METADATA_DIR, "signals_stats", "dict_signals_stats.yaml")
DEFAULT_SIGNALS_MEAN_STD_TRAIN_FILE = os_join(METADATA_DIR, "signals_stats", "dict_signals_mean_std_train.yaml")

DEFAULT_SHOTS_STATS_TEST_FILE = os_join(METADATA_DIR, "shots_stats", "shots_stats_test.csv")
DEFAULT_SHOTS_STATS_TRAIN_FILE = os_join(METADATA_DIR, "shots_stats", "shots_stats_train.csv")
DEFAULT_SHOTS_STATS_VAL_FILE = os_join(METADATA_DIR, "shots_stats", "shots_stats_val.csv")

DEFAULT_METADATA_OUTLIER_FILE = os_join(METADATA_DIR, "dict_outlier_metadata.yaml")
DEFAULT_TOKAMARK_DATA_SPLITS_FILE = os_join(METADATA_DIR, "TokaMark_data_splits.csv")

DEFAULT_CONFIG_GET_METADATA_FILE = os_join(SCRIPTS_DIR, "preprocessing", "config_get_metadata.yaml")
DEFAULT_CONFIG_GET_METADATA_DEMO_FILE = os_join(SCRIPTS_DIR, "preprocessing", "config_get_metadata_demo.yaml")
DEFAULT_CONFIG_MODEL_TEST_FILE = os_join(SCRIPTS_DIR, "config_model_test.yaml")
DEFAULT_CONFIG_MODEL_TEST_DEMO_FILE = os_join(SCRIPTS_DIR, "config_model_test_demo.yaml")
DEFAULT_CONFIG_TASK_TEST_FILE = os_join(SCRIPTS_DIR, "config_task_test.yaml")


# ======================================================================================================================
# Default values

DEFAULT_LOCAL_FLAG_VALUE = False
DEFAULT_BASE_FSSPEC_PROTOCOL = "simplecache"
DEFAULT_TARGET_FSSPEC_PROTOCOL = "s3"
DEFAULT_S3_ENDPOINT_URL = "https://s3.echo.stfc.ac.uk"
DEFAULT_S3_MAST_DATASET_PATH = "/mast/tokamark/v1"  # Other options: "mast/leve2/shots", "mast/test/leve2/shots"
DEFAULT_BASE_LOCAL_ZARR_PATH = "/mast/tokamark/v1"
