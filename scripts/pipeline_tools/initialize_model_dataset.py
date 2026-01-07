from Task_Model_Wrapper import TaskModelTransformWrapper

def initialize_model_dataset(
    dataset: Optional[Dataset],
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