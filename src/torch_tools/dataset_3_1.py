import torch
from torch.utils.data import Dataset

from tokamark.tasks import get_task_config, get_task_metadata
from tokamark.data import (
    initialize_MAST_dataset,
    initialize_TokaMark_dataset,
)

TASK_NAME = "task_3-1"
SHOT_IDS = [27938]
ZARR_PATH = "/mnt/data2/datasets/tokamark/data"


class TokaMarkTorchDataset(Dataset):
    """
    Convert a TokaMark iterable dataset into a PyTorch map-style Dataset.

    Each item contains:
        - shot_id
        - window_index
        - t_cut
        - input diagnostic signals
        - actuator signals
        - target signals
    """

    def __init__(self, tokamark_dataset):
        # TokaMarkDataset is iterable but does not implement __len__.
        # Materializing the windows allows indexing and shuffling.
        self.windows = list(tokamark_dataset)

        if not self.windows:
            raise ValueError("The TokaMark dataset contains no windows.")

        print("Number of materialized windows:", len(self.windows))

    def __len__(self):
        return len(self.windows)

    @staticmethod
    def _to_tensor(values):
        return torch.as_tensor(values, dtype=torch.float32)

    def __getitem__(self, index):
        window = self.windows[index]

        inputs = {
            name: self._to_tensor(signal["values"])
            for name, signal in window["input"].items()
        }
        actuators = {
            name: self._to_tensor(signal["values"])
            for name, signal in window["actuator"].items()
        }
        targets = {
            name: self._to_tensor(signal["values"])
            for name, signal in window["output"].items()
        }

        return {
            "shot_id": torch.tensor(window["shot_id"], dtype=torch.long),
            "window_index": torch.tensor(window["window_index"], dtype=torch.long),
            "t_cut": torch.tensor(window["t_cut"], dtype=torch.float32),
            "input": inputs,
            "actuator": actuators,
            "target": targets,
        }


def build_dataset(
    task_name: str = TASK_NAME,
    shot_ids=None,
    zarr_path: str = ZARR_PATH,
) -> TokaMarkTorchDataset:
    """Build the task_3-1 TokaMark PyTorch dataset end to end."""
    shot_ids = shot_ids or SHOT_IDS

    task_config = get_task_config(task_name)
    print("Task:", task_name)
    print("Task type:", task_config["task_type"])
    print("Input length:", task_config["task_window_segmenter"]["input_length"])
    print("Output length:", task_config["task_window_segmenter"]["output_length"])

    mast_dataset = initialize_MAST_dataset(
        config_task=task_config,
        shots_list=shot_ids,
        local_flag=True,
        use_std_scaling=False,
        remove_outliers=False,
        remove_bad_efit_rating=False,
        store_manager_settings={"base_local_zarr_path": zarr_path},
    )
    print("\nNumber of loaded shots:", len(mast_dataset))

    tokamark_dataset = initialize_TokaMark_dataset(
        dataset=mast_dataset,
        task_metadata=get_task_metadata(task_config),
        config_metadata=task_config,
        shuffle_windows=False,
    )

    return TokaMarkTorchDataset(tokamark_dataset)