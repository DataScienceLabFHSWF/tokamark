"""
Docstring reference: https://numpydoc.readthedocs.io/en/latest/format.html
Python style reference: https://google.github.io/styleguide/pyguide.html
"""

import os.path
from pathlib import Path
import pandas as pd
import numpy as np
from torch import Tensor as TorchTensor

from MAST_tools.constants import PROJECT_ROOT_DIR
from MAST_benchmark.tasks import GROUP_TASKS, TASKS_CONFIGS_MAP, get_signals_metadata
from MAST_benchmark.tools.utils import AutoAppendingDataFrame


ID_COLUMNS = ["shot_id", "window_index", "feature_name"]
METRIC_COLUMNS = ["RMSE", "MAE"]
COLUMNS = ID_COLUMNS + METRIC_COLUMNS  # FIXME: Pick better name [Rodrigo]

WINDOW_METRICS_FILENAME = "windows_metrics.csv"  # TODO: Perhaps move to constants (also others below) [Rodrigo]
SIGNAL_METRICS_FILENAME = "signals_metrics.csv"
TASK_METRICS_FILENAME   = "tasks_metrics.csv"   # noqa (ignore multiple spaces)
GROUP_METRICS_FILENAME  = "groups_metrics.csv"  # noqa (ignore multiple spaces)


# ======================================================================================================================
class WindowMetricsWriter:
    """
    TODO: Add docstrings [Rodrigo]
    """

    # ------------------------------------------------------------------------------------------------------------------
    def __init__(
            self,
            task: str,
            output_dir: str
    ) -> None:
        """
        Initialize class attributes.

        Parameters
        ----------
        task : str
            Input task.

        output_dir : str
            Output directory.

        Returns
        -------
        None

        """

        metrics_path = Path(output_dir) / task / WINDOW_METRICS_FILENAME
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        if metrics_path.exists():
            metrics_path.unlink()

        self.writer = AutoAppendingDataFrame(path=metrics_path)

    # ------------------------------------------------------------------------------------------------------------------
    def compute_and_append(
            self,
            y_target: np.ndarray,
            y_pred: np.ndarray,
            shot_ids: TorchTensor,
            window_indices: TorchTensor,
            feature_name: str,
            verbose: bool = False
    ) -> None:
        """
        Computation and column-stacking of RMSE and MAE metrics for a given input (y_target, y_pred) pair.

        Parameters
        ----------
        y_target : np.ndarray
            Input target data in np.ndarray format.
        y_pred : np.ndarray
            Input predicted data in np.ndarray format.
        shot_ids : TorchTensor
            Torch tensor with shot IDs.
        window_indices : TorchTensor
            Torch tensor with window indices.
        feature_name : str
            Name of target feature.
        verbose : bool
            If True, verbose mode is activated.
            Default: False.

        Returns
        -------
        None

        """

        rmse_per_sample = np.sqrt(np.mean((y_target - y_pred) ** 2, axis=1))
        mae_per_sample = np.mean(np.abs(y_target - y_pred), axis=1)

        data = np.column_stack(
            tup=(
                shot_ids,
                window_indices,
                [feature_name] * len(shot_ids),
                rmse_per_sample,
                mae_per_sample
            )
        )

        if verbose:
            if not data.any():
                print(f"WARNING: Empty data.")

        df_eval = pd.DataFrame(data=data, columns=COLUMNS)
        self.writer.append(df_rows=df_eval)


# ----------------------------------------------------------------------------------------------------------------------
def aggregate_windows_metrics(
        df: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Method for the aggregation of windows metrics (RMSE, NRMSE, NMAE, MAE).

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with windows metrics.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        Tuple of aggregated metrics in pd.DataFrame format.

    """

    # Compute signal-level score within each shot
    if "RMSE" in df.columns:
        df["RMSE"] = df["RMSE"]**2
    df_signals_shots = (
        df
        .drop(columns="window_index")
        .groupby(by=["shot_id", "feature_name"])
        .mean()
        .reset_index()
    )

    if "RMSE" in df_signals_shots.columns:
        df_signals_shots["RMSE"] = df_signals_shots["RMSE"]**0.5

    # Normalize signals per shot
    signal_std = get_signals_metadata()
    std = df_signals_shots["feature_name"].apply(lambda x: signal_std[x]["std"])
    df_signals_shots["NRMSE"] = df_signals_shots["RMSE"] / std
    df_signals_shots["NMAE"] = df_signals_shots["MAE"] / std

    # Average signals scores across shots
    df_signals = (
        df_signals_shots
        .drop(columns="shot_id")
        .groupby(by=["feature_name"])
        .mean()
    )

    # Compute task-level score within each shot
    df_task_shots = (
        df_signals_shots
        .drop(columns=["feature_name", "RMSE", "MAE"])
        .groupby(by=["shot_id"])
        .mean()
    )

    return df_signals, df_task_shots


# ----------------------------------------------------------------------------------------------------------------------
def compute_task_metrics(
        task: str = "task_1-1",
        output_dir: str = ".",
        save: bool = True
) -> pd.DataFrame:
    """
    Compute task-level metrics for target task.

    Parameters
    ----------
    task : str
        Target task.
        Optional. Default: "task_1-1".
    output_dir : str
        Taget output directory.
        Optional. Default: ".".
    save : bool
        If True, save metrics as pandas dataframe in the provided `output_dir` directory.
        Optional. Default: True.

    Returns
    -------
    pd.DataFrame
        Tasks metrics in pandas dataframe format.

    """

    if task not in TASKS_CONFIGS_MAP:
        print(f"WARNING: Task {task} is not known. Available tasks are: {str(TASKS_CONFIGS_MAP.keys())}")

    output_dir = Path(output_dir)
    target_file_path = output_dir/task/WINDOW_METRICS_FILENAME
    if not os.path.isfile(target_file_path):
        print(f"WARNING: Windows metric file {target_file_path} for {task} not found.")
        return pd.DataFrame([])  # TODO: Test if this works [Rodrigo]

    df = pd.read_csv(target_file_path)

    # Compute signal- and task-level score within each shot
    df_signals, df_task_shots = aggregate_windows_metrics(df=df)

    # Average task scores across shots
    df_task = df_task_shots.mean()

    # Append task-level scores to signal scores 
    df_signals.loc[task] = df_task

    if save:
        df_signals.to_csv(output_dir/task/TASK_METRICS_FILENAME)

    return df_signals


# ----------------------------------------------------------------------------------------------------------------------
def compute_all_metrics(
        output_dir: str = ".",
        save_locally: bool = True
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute task-level and group-level metrics for all tasks.

    Parameters
    ----------
    output_dir : str
        Target output directory.
        Optional. Default: ".".
    save_locally: bool
        If True, save metrics as pandas dataframes in the provided `output_dir` directory.
        Optional. Default: True.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        Signals and groups metrics in pandas dataframe format, respectively.

    """

    output_dir = Path(output_dir)

    all_signals = []
    all_tasks = []
    all_groups = []

    for group_id in GROUP_TASKS:
        for task in GROUP_TASKS[group_id]:
            file_path = output_dir/task/WINDOW_METRICS_FILENAME
            if not file_path.exists():
                print(f"WARNING: Task {task} was not evaluated, the corresponding files were not found.")
                continue

            df = pd.read_csv(file_path)

            # Compute signal- and task-level score within each shot
            df_signals, df_task_shots = aggregate_windows_metrics(df)

            df_signals["task"] = task
            all_signals.append(df_signals.reset_index())

            df_task_shots["task"] = task
            all_tasks.append(df_task_shots.reset_index())

        if len(all_tasks) > 0:
            df_tasks_shots = pd.concat(all_tasks)
            all_tasks = []

            df_tasks = (
                df_tasks_shots
                .drop(columns="shot_id")
                .groupby(by=["task"])
                .mean()
                .reset_index()
            )

            df_group = (
                df_tasks
                .drop(columns="task")
                .mean()
            )
            df_group = df_group.to_frame().T
            df_group["task"] = f"group_{group_id}"

            all_groups.append(df_group)
            all_groups.append(df_tasks)

    df_signals = pd.DataFrame()
    df_groups = pd.DataFrame()
    if len(all_signals) > 0:
        df_signals = pd.concat(all_signals)
        ordered_cols = [df_signals.columns[-1]] + df_signals.columns[:-1].to_list()
        df_signals = df_signals[ordered_cols]

        df_groups = pd.concat(all_groups)
        ordered_cols = [df_groups.columns[-1]] + df_groups.columns[:-1].to_list()
        df_groups = df_groups[ordered_cols]

    if save_locally:
        df_signals.to_csv(output_dir/SIGNAL_METRICS_FILENAME, index=False)
        df_groups.to_csv(output_dir/GROUP_METRICS_FILENAME, index=False)

    return df_signals, df_groups


# ======================================================================================================================
if __name__ == "__main__":

    # NOTE: Change the path to the outputs dir here:
    output_dir_ = f"{PROJECT_ROOT_DIR}/output/demo"  # FIXME: Test this [Rodrigo]

    compute_task_metrics(
        task="task_2-1",  # "task_2-1", "task_4-4"
        output_dir=output_dir_
    )

    compute_all_metrics(output_dir=output_dir_)

    print("DONE")
