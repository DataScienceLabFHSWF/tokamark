import torch
from torch.utils.data import DataLoader

from .dataset_3_1 import build_dataset

BATCH_SIZE = 32


def build_dataloader(
    batch_size: int = BATCH_SIZE,
    shuffle: bool = True,
    num_workers: int = 0,
    seed: int = 42,
) -> DataLoader:
    """Build the DataLoader for the task_3-1 TokaMark dataset."""
    torch_dataset = build_dataset()

    generator = torch.Generator()
    generator.manual_seed(seed)

    return DataLoader(
        dataset=torch_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
        generator=generator,
    )