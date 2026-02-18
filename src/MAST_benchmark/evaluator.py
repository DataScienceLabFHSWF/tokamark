from pathlib import Path
import pandas as pd
import numpy as np

from MAST_benchmark.tasks import group_tasks, tasks_configs_map, get_signals_metadata
from MAST_benchmark.tools.utils import AutoAppendingDataFrame


ID_COLUMNS = ['shot_id', 'window_id', 'feature_name']
METRIC_COLUMNS = ['RMSE', 'MAE']
COLUMNS = ID_COLUMNS+METRIC_COLUMNS

WINDOW_METRICS_FILE = 'windows_metrics.csv'
SIGNAL_METRICS_FILE = 'signals_metrics.csv'
TASK_METRICS_FILE   = 'tasks_metrics.csv'
GROUP_METRICS_FILE  = 'groups_metrics.csv'


class WindowMetricsWriter():
    
    def __init__(self, task, output_dir):

        metrics_path = Path(output_dir) / task / WINDOW_METRICS_FILE
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        if metrics_path.exists():
            metrics_path.unlink()

        self.writer = AutoAppendingDataFrame(metrics_path)

    def compute_and_append(self, y_target, y_pred, shot_id, window_id, feature_name):
        rmse_per_sample = np.sqrt(np.mean((y_target - y_pred) ** 2, axis=1))
        mae_per_sample  = np.mean(np.abs(y_target - y_pred), axis=1)

        data = np.column_stack((shot_id, 
                                window_id, 
                                [feature_name] * len(shot_id), 
                                rmse_per_sample,
                                mae_per_sample))
        df_eval = pd.DataFrame(data, columns=COLUMNS)
        self.writer.append(df_eval)


def aggregate_windows_metrics(df):
    # Compute signal level score within each shot
    if 'RMSE' in df.columns:
        df['RMSE'] = df['RMSE']**2
    df_signals_shots = (
        df
        .drop(columns='window_id')
        .groupby(by=['shot_id', 'feature_name'])
        .mean()
        .reset_index()
    )
    if 'RMSE' in df_signals_shots.columns:
        df_signals_shots['RMSE'] = df_signals_shots['RMSE']**0.5

    # Normalise signals per shot
    signal_std = get_signals_metadata()
    std = df_signals_shots['feature_name'].apply(lambda x: signal_std[x]['std'])
    df_signals_shots['NRMSE'] = df_signals_shots['RMSE'] / std
    df_signals_shots['NMAE'] = df_signals_shots['MAE'] / std

    # Average signals scores across shots
    df_signals = (
        df_signals_shots
        .drop(columns='shot_id')
        .groupby(by=['feature_name'])
        .mean()
    )

    # Compute task level score within each shot
    df_task_shots = (
        df_signals_shots
        .drop(columns=['feature_name', 'RMSE', 'MAE'])
        .groupby(by=['shot_id'])
        .mean()
    )

    return df_signals, df_task_shots


def compute_task_metrics(task='task_1-1', output_dir='.', save=True):
    if task not in tasks_configs_map:
        print(f'Warning: task {task} is not known. Available tasks are: ', str(tasks_configs_map.keys()))

    output_dir = Path(output_dir)
    df = pd.read_csv(output_dir/task/WINDOW_METRICS_FILE)

    # Compute signal and task level score within each shot
    df_signals, df_task_shots = aggregate_windows_metrics(df)

    # Average task scores across shots
    df_task = df_task_shots.mean()

    # Append task-level scores to signal scores 
    df_signals.loc[task] = df_task

    if save:
        df_signals.to_csv(output_dir/task/TASK_METRICS_FILE)

    return df_signals

def compute_all_metrics(output_dir='.', save_locally=True):
    output_dir = Path(output_dir)

    all_signals = []
    all_tasks = []
    all_groups = []

    for group_id in group_tasks:
        for task in group_tasks[group_id]:
            file_path = output_dir/task/WINDOW_METRICS_FILE
            if not file_path.exists():
                print(f'Warning: task {task} was yet evaluated, the corresponding files were not found.')
                continue

            df = pd.read_csv(file_path)

            # Compute signal and task level score within each shot
            df_signals, df_task_shots = aggregate_windows_metrics(df)

            df_signals['task'] = task
            all_signals.append(df_signals.reset_index())

            df_task_shots['task'] = task
            all_tasks.append(df_task_shots.reset_index())

        if len(all_tasks) > 0:
            df_tasks_shots = pd.concat(all_tasks)
            all_tasks = []

            df_tasks = (
                df_tasks_shots
                .drop(columns='shot_id')
                .groupby(by=['task'])
                .mean()
                .reset_index()
            )

            df_group = (
                df_tasks
                .drop(columns='task')
                .mean()
            )
            df_group = df_group.to_frame().T
            df_group['task'] = f'group_{group_id}'

            all_groups.append(df_group)
            all_groups.append(df_tasks)

    if len(all_signals) > 0:
        df_signals = pd.concat(all_signals)
        ordered_cols = [df_signals.columns[-1]] + df_signals.columns[:-1].to_list()
        df_signals = df_signals[ordered_cols]

        df_groups = pd.concat(all_groups)
        ordered_cols = [df_groups.columns[-1]] + df_groups.columns[:-1].to_list()
        df_groups = df_groups[ordered_cols]

        if save_locally:
            df_signals.to_csv(output_dir/SIGNAL_METRICS_FILE, index=False)
            df_groups.to_csv(output_dir/GROUP_METRICS_FILE, index=False)

        return df_signals, df_groups


if __name__ == "__main__":
    # NOTE: change the path to the outputs dir here:
    output_dir='/home/ir-zaya1/fusion/fairmast-data-preprocessing/output/'

    compute_task_metrics('task_4-4', output_dir)
    compute_all_metrics(output_dir)

    print('DONE')
