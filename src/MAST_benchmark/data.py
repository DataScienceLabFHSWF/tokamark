"""
Docstring reference: https://numpydoc.readthedocs.io/en/latest/format.html
Python style reference: https://google.github.io/styleguide/pyguide.html
"""

from typing import Optional, Mapping, Any
# import os
# import pickle
# import numpy as np

from MAST_tools.MAST_dataset import MastDataset
from MAST_benchmark.tools.TokaMark_dataset import TokaMarkDataset
from MAST_benchmark.tools.path import METADATA_DIR
from MAST_benchmark.tools.MAST_composite_transform import (
    build_common_signal_transform_map,
)


# ----------------------------------------------------------------------------------------------------------------------
def initialize_MAST_dataset(
    config_task: Mapping[str, Any],
    shots_list: list,
    local_flag: bool = True,
    use_std_scaling: bool = True,
    return_incomplete_shots: bool = True,
    remove_outliers: bool = True,
    verbose: bool = False
):
    """
    Initialize MAST dataset.

    Parameters
    ----------
    config_task : Mapping[str, Any],
        Task configuration dictionary.
    shots_list : list
        List of target shots.
    local_flag : bool
        If True, local mode is used.
    use_std_scaling : bool
        If True, standard scaling is used.
    return_incomplete_shots :
        If True, incomplete shots are allowed.
    remove_outliers : bool
        If True, outliers are removed.
    verbose : bool
        If True, verbose mode is activated.

    """

    # ..................................................................................................................
    # Get unique source-signal
    source_signal_list = (
        (config_task["sources_and_signals"].get("input_name") or [])
        + (config_task["sources_and_signals"].get("actuator_name") or [])
        + (config_task["sources_and_signals"].get("output_name") or [])
    )
    source_signal_list = [
        s for i, s in enumerate(source_signal_list) if s not in source_signal_list[:i]
    ]  # Unicity  

    # ..................................................................................................................
    # Create common transform map      
    signal_transform_map = build_common_signal_transform_map(
        source_signal_list, use_std_scaling
    )

    MAST_dataset = MastDataset(
        local=local_flag,
        shots_list=shots_list,
        source_signal_list=source_signal_list,
        signal_level_transform_map=signal_transform_map,
        shot_level_transform=None,
        return_incomplete_shots=return_incomplete_shots,
        remove_outliers=remove_outliers,
        verbose=verbose
    )

    return MAST_dataset


# ----------------------------------------------------------------------------------------------------------------------
def initialize_TokaMark_dataset(
    dataset: Optional[MastDataset],
    dict_task_metadata: Mapping[str, Any],
    config_task: Mapping[str, Any],
    model_specific_transform: Optional[Any] = None,
    test_mode: bool = False,
    shuffle_windows = True,
    shuffle_buffer_size = 512,
    *,
    verbose: bool = False,
) -> Optional[TokaMarkDataset]:
    """
COmment to add

    """

    if dataset is None:
        return None

    return TokaMarkDataset(
        base_dataset=dataset,
        dict_task_metadata=dict_task_metadata,
        config_task=config_task,
        model_transform=model_specific_transform,
        test_mode=test_mode,
        verbose=verbose,
        shuffle_windows=shuffle_windows,
        shuffle_buffer_size = shuffle_buffer_size
    )