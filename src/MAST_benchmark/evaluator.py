"""Benchmark evaluation utilities for MAST tasks.

Typical usage:
1. During inference, create ``WindowMetricsAccumulator(task)`` and call
   ``add_batch(...)`` for each model batch.
2. After one task is complete, call:
   ``compute_metrics(task, output_dir, window_metrics_accumulator, ...)``.
   Default save flags are:
   - ``save_windows_metrics=False``
   - ``save_shot_metrics=False``
   - ``save_task_metrics=True``
   With defaults, only ``<output_dir>/<task>/task_metrics.csv`` is saved.
   To enable other files, pass:
   - ``save_windows_metrics=True`` for ``<output_dir>/<task>/windows_metrics.csv``
   - ``save_shot_metrics=True`` for ``<output_dir>/<task>/shots_metrics.csv``
   (you can enable any combination of the three flags).
3. After all tasks are complete, call
   ``compute_summary_metrics(output_dir, source=...)`` to produce
   ``signals_metrics.csv`` and ``groups_metrics.csv`` in ``output_dir``.
   Supported ``source`` values are:
   - ``source="task_metrics"`` (default): reads ``task_metrics.csv`` per task
   - ``source="windows_metrics"``: reads ``windows_metrics.csv`` per task
   - ``source="shots_metrics"``: reads ``shots_metrics.csv`` per task

Aggregation follows the benchmark hierarchy:
- signal rows: equal-weight mean/std across shots
- task rows: equal-weight mean/std across shots
- group rows: equal-weight mean of task means and mean of task stds
"""

from pathlib import Path
import pandas as pd
import numpy as np

from MAST_benchmark.tasks import group_tasks, tasks_configs_map, get_signals_metadata


ID_COLUMNS = ["shot_id", "window_index", "feature_name"]
METRIC_COLUMNS = ["RMSE", "MAE"]
COLUMNS = ID_COLUMNS + METRIC_COLUMNS
TASK_METRICS = ("NRMSE", "NMAE", "RMSE", "MAE")
TASK_SUMMARY_COLUMNS = ("n_shots",) + tuple(
    f"{metric}_{suffix}" for metric in TASK_METRICS for suffix in ("mean", "std_pop")
)
SIGNAL_STD_COLUMNS = tuple(f"{metric}_std_pop" for metric in TASK_METRICS)
SIGNAL_VALUE_COLUMNS = TASK_METRICS

WINDOW_METRICS_FILE = "windows_metrics.csv"
SHOT_METRICS_FILE = "shots_metrics.csv"
SIGNAL_METRICS_FILE = "signals_metrics.csv"
TASK_METRICS_FILE = "task_metrics.csv"
GROUP_METRICS_FILE = "groups_metrics.csv"


def compute_windows_metrics(y_target, y_pred, shot_id, window_index, feature_name):
    """Compute RMSE/MAE per window for one feature and return a dataframe chunk."""
    rmse_per_sample = np.sqrt(np.mean((y_target - y_pred) ** 2, axis=1))
    mae_per_sample = np.mean(np.abs(y_target - y_pred), axis=1)

    # Build columns explicitly to preserve numeric dtypes (avoid mixed-type upcast).
    return pd.DataFrame(
        {
            "shot_id": np.asarray(shot_id),
            "window_index": np.asarray(window_index),
            "feature_name": np.asarray([feature_name] * len(rmse_per_sample), dtype=object),
            "RMSE": np.asarray(rmse_per_sample, dtype=float),
            "MAE": np.asarray(mae_per_sample, dtype=float),
        }
    )


class WindowMetricsAccumulator:
    """In-memory collector of per-window metrics for one task."""

    def __init__(self, task):
        self.task = str(task)
        self._chunks: list[pd.DataFrame] = []

    def add_batch(self, y_target, y_pred, shot_id, window_index, feature_name):
        """Compute one metrics chunk and append it to the accumulator."""
        self._chunks.append(
            compute_windows_metrics(
                y_target=y_target,
                y_pred=y_pred,
                shot_id=shot_id,
                window_index=window_index,
                feature_name=feature_name,
            )
        )

    def is_empty(self):
        """Return True when no window chunks were collected."""
        return len(self._chunks) == 0

    def to_dataframe(self):
        """Concatenate all collected chunks into one windows dataframe."""
        if self.is_empty():
            return pd.DataFrame(columns=COLUMNS)
        return pd.concat(self._chunks, ignore_index=True)


def _build_task_metrics_df(task, df_signals, df_task_shots):
    """Build the task dataframe (signal rows + one task summary row)."""
    df_task_metrics = df_signals.reset_index().rename(columns={"index": "feature_name"})
    df_task_metrics["feature_name"] = df_task_metrics["feature_name"].astype(str)

    summary = _build_task_summary_row_from_shots(df_task_shots)
    task_row_df = pd.DataFrame([{"feature_name": task, **summary}])

    for col in TASK_SUMMARY_COLUMNS:
        if col not in df_task_metrics.columns:
            df_task_metrics[col] = np.nan

    df_task_metrics = pd.concat([df_task_metrics, task_row_df], ignore_index=True)

    ordered_cols = ["feature_name"] + list(TASK_SUMMARY_COLUMNS)
    remaining_cols = [c for c in df_task_metrics.columns if c not in ordered_cols]
    return df_task_metrics[ordered_cols + remaining_cols]


def _build_task_summary_row_from_shots(df_task_shots):
    """Build one task-level summary row from shot-level metrics."""
    summary = {"n_shots": float(len(df_task_shots))}
    for metric in TASK_METRICS:
        if metric in df_task_shots.columns:
            vals = pd.to_numeric(df_task_shots[metric], errors="coerce").dropna()
        else:
            vals = pd.Series(dtype=float)

        if len(vals) == 0:
            mean_val = np.nan
            std_val = np.nan
        else:
            mean_val = float(vals.mean())
            std_val = float(vals.std(ddof=0))

        summary[f"{metric}_mean"] = mean_val
        summary[f"{metric}_std_pop"] = std_val
    return summary


def _build_group_summary_from_task_summaries(df_tasks):
    """Build one group summary row from equal-weight task means/stds."""
    group = {
        "n_shots": float(pd.to_numeric(df_tasks["n_shots"], errors="coerce").sum())
    }

    for metric in TASK_METRICS:
        mean_col = f"{metric}_mean"
        std_col = f"{metric}_std_pop"
        if mean_col not in df_tasks.columns:
            group[mean_col] = np.nan
            group[std_col] = np.nan
            continue

        tmp = df_tasks[[mean_col, std_col]].copy()
        tmp[mean_col] = pd.to_numeric(tmp[mean_col], errors="coerce")
        tmp[std_col] = pd.to_numeric(tmp[std_col], errors="coerce")
        tmp = tmp.dropna()

        if len(tmp) == 0:
            group[mean_col] = np.nan
            group[std_col] = np.nan
            continue

        means = tmp[mean_col].to_numpy(dtype=float)
        stds = tmp[std_col].to_numpy(dtype=float)

        group[mean_col] = float(np.mean(means))
        group[std_col] = float(np.mean(stds))

    return group


def _extract_task_summary_from_task_metrics(task, output_dir):
    """Load one task summary row (and signal rows) from ``task_metrics.csv``."""
    file_path = Path(output_dir) / task / TASK_METRICS_FILE
    if not file_path.exists():
        print(
            f"Warning: task {task} was yet evaluated, the corresponding files were not found."
        )
        return None, None

    df = pd.read_csv(file_path)
    if "feature_name" not in df.columns:
        print(f"Warning: {file_path} has no feature_name column. Skipping task {task}.")
        return None, None

    df["feature_name"] = df["feature_name"].astype(str)
    task_rows = df[df["feature_name"] == task]
    if len(task_rows) == 0:
        print(f"Warning: {file_path} has no task row '{task}'. Skipping task.")
        return None, None

    row = task_rows.iloc[0]
    summary = {"task": task}
    for col in TASK_SUMMARY_COLUMNS:
        summary[col] = row[col] if col in row.index else np.nan

    signal_rows = df[df["feature_name"] != task].copy()
    for col in SIGNAL_STD_COLUMNS:
        if col not in signal_rows.columns:
            signal_rows[col] = np.nan

    ordered_signal_cols = (
        ["feature_name"]
        + [c for c in SIGNAL_VALUE_COLUMNS if c in signal_rows.columns]
        + [c for c in SIGNAL_STD_COLUMNS if c in signal_rows.columns]
    )
    signal_rows = signal_rows[ordered_signal_cols]
    signal_rows["task"] = task
    return summary, signal_rows


def _extract_task_summary_from_shots_metrics(task, output_dir):
    """Load one task summary row from ``shots_metrics.csv``."""
    file_path = Path(output_dir) / task / SHOT_METRICS_FILE
    if not file_path.exists():
        print(
            f"Warning: task {task} was yet evaluated, the corresponding files were not found."
        )
        return None

    df = pd.read_csv(file_path)
    if "shot_id" not in df.columns:
        print(f"Warning: {file_path} has no shot_id column. Skipping task {task}.")
        return None

    df_task_shots = df.copy()
    for metric in TASK_METRICS:
        if metric not in df_task_shots.columns:
            df_task_shots[metric] = np.nan
        df_task_shots[metric] = pd.to_numeric(df_task_shots[metric], errors="coerce")

    df_task_shots = df_task_shots.set_index("shot_id")
    return {"task": task, **_build_task_summary_row_from_shots(df_task_shots)}


def _extract_task_summary_from_windows_metrics(task, output_dir):
    """Load one task summary row (and signal rows) from ``windows_metrics.csv``."""
    file_path = Path(output_dir) / task / WINDOW_METRICS_FILE
    if not file_path.exists():
        print(
            f"Warning: task {task} was yet evaluated, the corresponding files were not found."
        )
        return None, None

    df = pd.read_csv(file_path)
    df_signals, df_task_shots = aggregate_windows_metrics(df)
    summary = {"task": task, **_build_task_summary_row_from_shots(df_task_shots)}
    signal_rows = df_signals.reset_index().assign(task=task)
    return summary, signal_rows


def aggregate_windows_metrics(df):
    """Aggregate window rows into signal-level and shot-level dataframes."""
    df = df.copy()
    # Be robust to string/object inputs from in-memory paths and CSV readers.
    df["RMSE"] = pd.to_numeric(df["RMSE"], errors="coerce")
    df["MAE"] = pd.to_numeric(df["MAE"], errors="coerce")
    df = df.dropna(subset=["RMSE", "MAE"])

    # Compute signal level score within each shot
    if "RMSE" in df.columns:
        df["RMSE"] = df["RMSE"] ** 2
    df_signals_shots = (
        df.drop(columns="window_index")
        .groupby(by=["shot_id", "feature_name"])
        .mean()
        .reset_index()
    )
    if "RMSE" in df_signals_shots.columns:
        df_signals_shots["RMSE"] = df_signals_shots["RMSE"] ** 0.5

    # Normalise signals per shot
    signal_std = get_signals_metadata()
    std = df_signals_shots["feature_name"].apply(lambda x: signal_std[x]["std"])
    df_signals_shots["NRMSE"] = df_signals_shots["RMSE"] / std
    df_signals_shots["NMAE"] = df_signals_shots["MAE"] / std

    # Average signals scores across shots
    df_signals_mean = (
        df_signals_shots.drop(columns="shot_id").groupby(by=["feature_name"]).mean()
    )
    df_signals_std = (
        df_signals_shots.drop(columns="shot_id")
        .groupby(by=["feature_name"])
        .std(ddof=0)
        .fillna(0.0)
        .rename(columns=lambda c: f"{c}_std_pop")
    )
    df_signals = df_signals_mean.join(df_signals_std)

    # Compute task level score within each shot
    df_task_shots = (
        df_signals_shots.drop(columns=["feature_name"]).groupby(by=["shot_id"]).mean()
    )

    return df_signals, df_task_shots


def compute_metrics(
    task,
    output_dir,
    window_metrics_accumulator,
    save_windows_metrics=False,
    save_shot_metrics=False,
    save_task_metrics=True,
):
    """Compute one task metrics from an in-memory window metrics accumulator."""
    if task not in tasks_configs_map:
        print(
            f"Warning: task {task} is not known. Available tasks are: ",
            str(tasks_configs_map.keys()),
        )

    output_dir = Path(output_dir)
    task_output_dir = output_dir / task
    task_output_dir.mkdir(parents=True, exist_ok=True)
    if window_metrics_accumulator is None:
        raise ValueError(
            "window_metrics_accumulator cannot be None in compute_metrics(...)"
        )

    if not isinstance(window_metrics_accumulator, WindowMetricsAccumulator):
        raise TypeError(
            "window_metrics_accumulator must be a WindowMetricsAccumulator instance"
        )

    if window_metrics_accumulator.task != str(task):
        raise ValueError(
            "Accumulator task does not match compute_metrics task: "
            f"{window_metrics_accumulator.task} vs {task}"
        )

    windows_df = window_metrics_accumulator.to_dataframe()
    if len(windows_df) == 0:
        raise ValueError(f"No window metrics were collected for task {task}.")

    missing = [c for c in COLUMNS if c not in windows_df.columns]
    if len(missing) > 0:
        raise ValueError(
            "windows_df is missing required columns: " + ", ".join(missing)
        )
    df = windows_df[COLUMNS].copy()

    # Compute signal and task level score within each shot
    df_signals, df_task_shots = aggregate_windows_metrics(df)
    df_task_metrics = _build_task_metrics_df(task, df_signals, df_task_shots)

    if save_windows_metrics:
        df.to_csv(task_output_dir / WINDOW_METRICS_FILE, index=False)
    if save_shot_metrics:
        df_shots = df_task_shots.reset_index()
        n_windows = (
            df.groupby("shot_id", as_index=False)["window_index"]
            .nunique()
            .rename(columns={"window_index": "n_windows"})
        )
        df_shots = df_shots.merge(n_windows, on="shot_id", how="left")
        ordered_cols = ["shot_id", "n_windows"] + [
            c for c in df_shots.columns if c not in {"shot_id", "n_windows"}
        ]
        df_shots = df_shots[ordered_cols]
        df_shots.to_csv(task_output_dir / SHOT_METRICS_FILE, index=False)
    if save_task_metrics:
        df_task_metrics.to_csv(task_output_dir / TASK_METRICS_FILE, index=False)

    return df_task_metrics.set_index("feature_name")


def compute_summary_metrics(output_dir=".", source="task_metrics", save_locally=True):
    """Aggregate all tasks into signal/group metrics from task/window/shot files."""
    output_dir = Path(output_dir)

    all_signals = []
    all_groups = []

    if source not in {"task_metrics", "windows_metrics", "shots_metrics"}:
        raise ValueError(
            "source must be one of: 'task_metrics', 'windows_metrics', 'shots_metrics'"
        )

    for group_id in group_tasks:
        group_task_summaries = []

        for task in group_tasks[group_id]:
            if source == "windows_metrics":
                task_summary, signal_rows = _extract_task_summary_from_windows_metrics(
                    task, output_dir
                )
                if task_summary is None:
                    continue
                group_task_summaries.append(task_summary)
                if signal_rows is not None and len(signal_rows) > 0:
                    all_signals.append(signal_rows)
            elif source == "shots_metrics":
                task_summary = _extract_task_summary_from_shots_metrics(task, output_dir)
                if task_summary is None:
                    continue
                group_task_summaries.append(task_summary)
            else:
                task_summary, signal_rows = _extract_task_summary_from_task_metrics(
                    task, output_dir
                )
                if task_summary is None:
                    continue
                group_task_summaries.append(task_summary)
                if signal_rows is not None and len(signal_rows) > 0:
                    all_signals.append(signal_rows)

        if len(group_task_summaries) > 0:
            df_tasks = pd.DataFrame(group_task_summaries)
            df_group_values = _build_group_summary_from_task_summaries(df_tasks)
            df_group_row: dict[str, object] = {"task": f"group_{group_id}"}
            df_group_row.update(df_group_values)

            all_groups.append(pd.DataFrame([df_group_row]))
            all_groups.append(df_tasks)

    df_signals = (
        pd.concat(all_signals, ignore_index=True)
        if len(all_signals) > 0
        else pd.DataFrame()
    )
    if len(df_signals) > 0 and "task" in df_signals.columns:
        ordered_cols = ["task"] + [c for c in df_signals.columns if c != "task"]
        df_signals = df_signals[ordered_cols]

    df_groups = (
        pd.concat(all_groups, ignore_index=True)
        if len(all_groups) > 0
        else pd.DataFrame()
    )
    if len(df_groups) > 0 and "task" in df_groups.columns:
        ordered_cols = ["task"] + [c for c in df_groups.columns if c != "task"]
        df_groups = df_groups[ordered_cols]

    if save_locally:
        if len(df_signals) > 0:
            df_signals.to_csv(output_dir / SIGNAL_METRICS_FILE, index=False)
        if len(df_groups) > 0:
            df_groups.to_csv(output_dir / GROUP_METRICS_FILE, index=False)

    return df_signals, df_groups


if __name__ == "__main__":
    # NOTE: change the path to the outputs dir here:
    output_dir = "/home/ir-zaya1/fusion/fairmast-data-preprocessing/output/"

    compute_summary_metrics(output_dir)

    print("DONE")
