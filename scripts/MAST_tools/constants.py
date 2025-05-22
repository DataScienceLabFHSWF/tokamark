# Contributors:
# - Rodrigo Ordonez-Hurtado (rodrigo.ordonez.hurtado@ibm.com).
# Docstring reference: https://numpydoc.readthedocs.io/en/latest/format.html

from typing import Union
import zarr.storage
from xarray.core.dataset import Dataset

NoneType = type(None)
ZarrStoreType = zarr.storage.FsspecStore
DataSourceType = Union[dict, ZarrStoreType]
XarrayDatasetType = Dataset
