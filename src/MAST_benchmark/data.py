from typing import Optional, Mapping, Any
import os
import pickle
import numpy as np

from MAST_tools.MAST_dataset import MastDataset
from MAST_benchmark.tools.Task_Model_Wrapper import TaskModelTransformWrapper
from MAST_benchmark.tools.path import METADATA_DIR
from MAST_benchmark.tools.MAST_composite_transform import (
    build_common_signal_transform_map,
)


# ----------------------------------------------------------------------------------------------------------------------
def initialize_MAST_dataset(
    config_task,
    local_flag,
    shots_list,
    use_std_scaling = True,
    return_incomplete_shots=True
):

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
    )

    return MAST_dataset


# ----------------------------------------------------------------------------------------------------------------------
def initialize_model_dataset(
    dataset: Optional[MastDataset],
    dict_metadata: Mapping[str, Any],
    config_task: Mapping[str, Any],
    model_specific_transform=None,
    *,
    verbose: bool = False,
) -> Optional[Any]:
    """
    Wrap a single baseline shot-level dataset with TaskModelTransformWrapper.

    This is a single-split version of initialize_model_datasets(), intended to
    be called explicitly for each split (train/val/test) from entrypoint scripts.

    Parameters
    ----------
    dataset:
        Baseline shot-level dataset (e.g., MastDataset) for one split, or None.
    dict_metadata:
        Metadata dictionary produced by the baseline pipeline (dt, shapes, etc.).
    config_task:
        Task configuration dict containing `task_window_segmenter` (keys, lengths, delta).
    model_specific_transform:
        Optional model-specific transform chain applied per window.
    verbose:
        If True, enables verbose prints in the wrapper.

    Returns
    -------
    TaskModelTransformWrapper | None
        Wrapped dataset, or None if input dataset is None.
    """
    if dataset is None:
        return None

    return TaskModelTransformWrapper(
        dataset,
        dict_metadata,
        config_task,
        model_specific_transform,
        verbose=verbose,
    )