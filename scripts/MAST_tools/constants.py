"""
Docstring reference: https://numpydoc.readthedocs.io/en/latest/format.html
Python style reference: https://google.github.io/styleguide/pyguide.html
"""

from typing import Union
import zarr.storage
from xarray.core.dataset import Dataset

ZarrFSStoreType = zarr.storage.FsspecStore
ZarrLocalStoreType = zarr.storage.LocalStore
ZarrStoreType = Union[ZarrFSStoreType, ZarrLocalStoreType]
DataSourceType = Union[dict, ZarrStoreType]
XarrayDatasetType = Dataset
