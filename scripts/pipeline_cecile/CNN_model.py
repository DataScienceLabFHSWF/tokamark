import joblib
import json
import numpy as np
import os
import pandas as pd
import sys

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torch.utils.data.dataloader import default_collate


cwd = os.path.dirname(os.getcwd())
mother_dir = os.path.dirname(cwd) + os.sep
sys.path.append(os.path.abspath(os.path.join(mother_dir , "MAST_tools")))
sys.path.append(mother_dir)
sys.path.append(cwd)
sys.path.append(os.path.join( os.path.dirname(cwd) ) )



import torch
import torch.nn as nn
import torch.nn.functional as F

class SmallCNNBranch(nn.Module):
    def __init__(self, input_len, D):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.BatchNorm1d(input_len),
            nn.Conv1d(input_len, D, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(D),
            nn.MaxPool1d(2),
            nn.Conv1d(D, 2 * D, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(2 * D),
            nn.MaxPool1d(2),
            nn.Conv1d(2 * D, 4 * D, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(4 * D),
        )
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(4 * D, input_len)

    def forward(self, x):
        x = self.cnn(x)
        x = self.global_avg_pool(x).squeeze(-1)  # [B, C]
        x = self.fc(x)
        return x


class NumericalBranch(nn.Module):
    def __init__(self, input_len):
        super().__init__()
        self.bn = nn.BatchNorm1d(input_len)

    def forward(self, x):
        return self.bn(x)


class MultiBranchCNNModel(nn.Module):
    def __init__(self, input_shapes, output_shape, D=16):
        super().__init__()

        self.branches = nn.ModuleList()
        self.output_shape = output_shape[0][0]
        merged_dim = 0

        for shape in input_shapes:
            if len(shape) == 2:  # e.g., (1, 15)
                input_len = shape[0]
                branch = SmallCNNBranch(input_len, D)
                merged_dim += input_len
            elif len(shape) == 1:  # e.g., (1,)
                input_len = shape[0]
                branch = NumericalBranch(input_len)
                merged_dim += shape[0]
            else:
                raise ValueError(f"Unsupported input shape: {shape}")
            self.branches.append(branch)

        self.fc = nn.Sequential(
            nn.BatchNorm1d(merged_dim),
            nn.Linear(merged_dim, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Dropout(0.2),
            nn.Linear(32, self.output_shape),
        )

    def forward(self, *inputs):
        branch_outputs = []

        for branch, x in zip(self.branches, inputs):
            out = branch(x)
            branch_outputs.append(out)

        merged = torch.cat(branch_outputs, dim=1)
        return self.fc(merged)