import os

from MAST_benchmark.tools.path import TASKS_CONFIGS_DIR
from MAST_benchmark.tools.utils import get_config_from_yaml


tasks_configs_map = {
    "task_1-1": "group_1_reconstruction/task_1-1.yaml",
    "task_1-2": "group_1_reconstruction/task_1-2.yaml",
    "task_1-3": "group_1_reconstruction/task_1-3.yaml",
    "task_2-1": "group_2_magnetics_dynamics/task_2-1.yaml",
    "task_2-2": "group_2_magnetics_dynamics/task_2-2.yaml",
    "task_2-3": "group_2_magnetics_dynamics/task_2-3.yaml",
    "task_3-1": "group_3_profiles_dynamics/task_3-1.yaml",
    "task_3-2": "group_3_profiles_dynamics/task_3-2.yaml",
    "task_3-3": "group_3_profiles_dynamics/task_3-3.yaml",
    "task_4-1": "group_4_mhd_activity/task_4-1.yaml",
    "task_4-2": "group_4_mhd_activity/task_4-2.yaml",
    "task_4-3": "group_4_mhd_activity/task_4-3.yaml",
    "task_4-4": "group_4_mhd_activity/task_4-4.yaml",
    "task_4-5": "group_4_mhd_activity/task_4-5.yaml"
}

def get_task_config(task_name):
    task_path = tasks_configs_map[task_name]
    file_path = os.path.join(TASKS_CONFIGS_DIR, task_path)

    return get_config_from_yaml(file_path)
