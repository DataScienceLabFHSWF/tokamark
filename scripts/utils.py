import numpy as np
import os
import pandas as pd
import random
import sys
from typing import Optional, Any

cwd = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cwd)

from MAST_tools.signal_utils import MASTSignalManager  # noqa
from MAST_tools.store_utils import MASTStorageManager  # noqa


# ----------------------------------------------------------------------------------------------------------------------
def is_finite_numeric_array(
        arr: Any
) -> bool:
    """
    Check if input array is numeric and finite (i.e., does not contain NaN or Inf values).

    Parameters
    ----------
    arr : Any
        Input array to be checked.

    Returns
    -------
    bool
        Result of corresponding check.

    """

    # Check if array is numeric
    if not np.issubdtype(arr.dtype, np.number):
        return False

    # Check if all values are finite (no NaN or Inf)
    return np.isfinite(arr).all()


# ----------------------------------------------------------------------------------------------------------------------
def read_signals(
        filepath: str
) -> dict:
    """
    Read signals from given filepath.

    Parameters
    ----------
    filepath : str
        Path to file containing all signal names and their multiplicity (number of channels).

    Returns
    -------
    dict
        Dictionary with signal names as keys and number of traces as values. Empty dictionary if exception raised during
        reading.

    """

    signals = dict()
    try:
        with open(filepath, "r") as file:
            for line in file:
                signals[line.split()[0]] = int(line.split()[1])
    except FileNotFoundError:
        print(f"Error: File not found at '{filepath}'")
        return {}
    except Exception as e:
        print(f"An error occurred while reading the file '{filepath}': {e}")
        return {}
    return signals


# ----------------------------------------------------------------------------------------------------------------------
def shuffle_shot_ids(
        shot_ids: list[int],
        seed: Optional[int] = None
) -> list[int]:
    """
    Shuffle list of shot IDs.

    Parameters
    ----------
    shot_ids : list[int]
        Target list of shot IDs to be shuffled.
    seed : Optional[int]
        Seed value.
        Optional. Default: None.

    Returns
    -------
    list[int]
        Shuffled list of shot IDs.

    """

    random.seed(a=seed)
    random.shuffle(x=shot_ids)

    return shot_ids


# ----------------------------------------------------------------------------------------------------------------------
def make_dataframe_from_shot_ids(
        store_manager: MASTStorageManager,
        shot_ids: list[int],
        group: str,
        signal_name: str,
        local: bool = True,
        level: int = 2,
        test_data: bool = False
) -> tuple[list[Any] | None, list[Any]]:
    """
    Get a dataFrame from concatenated signals.

    Parameters
    ----------
    store_manager : MASTStorageManager
        Instance of the `MASTStorageManager` class.
    shot_ids : list[int]
        List of target shot IDs.
    group : str
        Target group (source) name.
    signal_name : str
        Target signal name.
    local : bool
        If True, the target shot is pulled from locally stored data (e.g., in the CSD3 cluster), otherwise it is pulled
        from the registered remote data repository (e.g., a cloud S3 bucket).
        Optional. Default: False.
    level : int
        Target level for the MAST data/metadata to be pulled.
        Optional. Default: 2.
    test_data : bool
        If True, the target shot is pulled from test data, otherwise it is pulled from curated data. Not available for
        locally stored data (i.e, if `local` is True).
        Optional. Default: False.

    Returns
    -------
    tuple[list[Any] | None, list[Any]]
        Tuple (channels, shot_list).

    """

    channels = None
    shot_list = []
    for shot_id in shot_ids:

        sig = MASTSignalManager()
        sig._set_store_manager(store_manager=store_manager)  # noqa

        store = sig.store_manager.make_shot_store(
            shot_info={"shot_id": shot_id, "level": level, "test_data": test_data, "local": local}
        )

        sig_values = sig.get_signal_values(
            data_origin=store,
            source_name=group,
            signal_name=signal_name
        )

        if channels is None:
            channels = list(sig.get_channel_names(signal_name=signal_name, data_origin=store, source_name=group))

        df = pd.DataFrame(sig_values.T, columns=channels)
        shot_list.append(df)
    
    return channels, shot_list


# ======================================================================================================================
if __name__ == "__main__":
    pass
