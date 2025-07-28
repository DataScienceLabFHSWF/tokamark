import argparse
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


cwd = os.path.dirname(os.path.abspath(__file__))
mother_dir = os.path.dirname(cwd) + os.sep
sys.path.append(mother_dir)

from MAST_tools.MAST_dataset import MastDataset_test as MastDataset
from MAST_tools.signal_utils import MASTSignalManager  
from MAST_tools.store_utils import MASTStorageManager

from pipelines.transforms.signal_level_transforms.imputer_transform import ImputerTransform
from pipelines.transforms.signal_level_transforms.pca_transform import PCATransform 
from pipelines.transforms.signal_level_transforms.compose_transform import ComposeTransform
from pipelines.transforms.signal_level_transforms.segmenter_transform import SegmenterTransform

from pipelines.collate_functions.collate_functions import MiniBatchCollateFn

from pipelines.configs.config_setup import get_settings

from  pipelines.utils.utils import ( load_models,
                          ComposeTransforms,
                          read_data_split_csv)

#================================================================
        ##########    DATALOADER  ##########
#================================================================
def initialize_dataset(
    local: bool,
    shots_list: list[int],
    source_signal_list: list[tuple[str, str]],
    signal_level_transform_map: dict = None,
    shot_level_transform: callable = None
):
    """Initialize the MASTDataset.
    Parameters
    ----------
    local : bool
        If True, use local MAST database, otherwise use remote S3 bucket
    shots_list : list[int]
        List of shot IDs to load data for.
    source_signal_list : list[tuple[str, str]]
        List of data names to load, format: ('source', 'signal')
    signal_level_transform_map : dict, optional
        dict map of pipeline of transforms to apply at signal level, by default None
    shot_level_transform : callable, optional
        pipeline of transforms to apply at shot level, by default None
    Returns
    -------
    MASTDataset
        Initialized MASTDataset object.
    """
    
    dataset = MastDataset(
        local=local,
        shots_list=shots_list,
        source_signal_list=source_signal_list,
        signal_level_transform_map=signal_level_transform_map,
        shot_level_transform=shot_level_transform
    )
    
    return dataset

def initialize_dataloader(
    dataset: MastDataset,
    batch_size: int,
    num_workers: int,
    shuffle: bool = True,
    collate_fn: callable = None
):
    """Initialize the DataLoader for the MASTDataset.
    
    Parameters
    ----------
    dataset : MASTDataset
        The dataset to load.
    batch_size : int
        Size of the batch.
    num_workers : int
        Number of workers for DataLoader.
    shuffle : bool, optional
        Whether to shuffle the data, by default True
    collate_fn : callable, optional
        Custom collate function, by default None
    
    Returns
    -------
    DataLoader
        Initialized DataLoader object.
    """
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        collate_fn=collate_fn
    )
    
    return dataloader


#================================================================
    ########## Model Definition and Training ##########
#================================================================
class NeuralNetwork(nn.Module):
    """
    Simple feedforward neural network.

    Architecture
        - Fully connected layer with 64 units
        - ReLU activation
        - Fully connected output layer with 2 units
    """

    def __init__(self, input_size, output_size, l1_size, l2_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, l1_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(l1_size, l2_size)
        self.fc3 = nn.Linear(l2_size, output_size)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def get_input_output_size(dataloader, device='cpu'):
    """
    Infers the input and output dimensions of the model from the first batch
    after flattening signals.

    Parameters
    ----------
    dataloader : torch.utils.data.DataLoader
        The training dataloader.
    device : str or torch.device
        The device to load the tensors on.

    Returns
    -------
    tuple[int, int]
        input_dim, output_dim
    """
    for batch in dataloader:
        x_batch = batch['x'] # List of lists 
        y_batch = batch['y'] # List of lists
        
        x_tensor_list = []
        y_tensor_list = []

        x_signals = x_batch[0]
        y_signals = y_batch[0]
    
        if not x_signals or not y_signals:
            raise ValueError("First segment in x_batch or y_batch is empty.")
        
        try:
            x_flat = torch.cat([sig.to(device).flatten() for sig in x_signals], dim=0)
            y_flat = torch.cat([sig.to(device).flatten() for sig in y_signals], dim=0)
        except Exception as e:
            raise ValueError(f"Failed to concatenate tensors: {e}")
        
        input_dim = x_flat.shape[0]
        output_dim = y_flat.shape[0]
        print(f"NN size:  input_dim = {input_dim}, output_dim = {output_dim }")

        return input_dim, output_dim
   
    
def compare_last_first_tensor_shapes(x_list,y_list):
    """ Check that the shape of the last element in the lists is coinsistent 
    with the number of signals expected, e.g., the nr_signals found in the first element

    Parameters
    ----------
    x_list : list[Tensor]
        Each entry in y_list is a list of tensors each one of shape :
        (nr_features, time length).
        Each one corresponding to a segment (or time-window) of the original input signals.
    y_list : list[Tensor]
        Each entry in y_list is a list of tensors each one of shape :
        (nr_features, time length)..
        Each one corresponding to a segment (or time-window) of the original target signals.

    Returns
    -------
    x_list, y_list minus the last tensors 
    if these have different shapes from the first tensor
    """

    if not x_list or not y_list:
        print("Warning: One or both lists are empty, returning as-is.")
        return x_list, y_list
    if len(x_list) <= 1 or len(y_list) <= 1:
        return x_list, y_list

    # Compare number of tensors (nr_signals) in first and last entries
    input_num_signals_0 = len(x_list[0])  # Number of tensors in first x_list entry
    output_num_signals_0 = len(y_list[0])  # Number of tensors in first y_list entry
    input_num_signals_1 = len(x_list[-1])  # Number of tensors in last x_list entry
    output_num_signals_1 = len(y_list[-1])  # Number of tensors in last y_list entry

    if input_num_signals_0 != input_num_signals_1 or output_num_signals_0 != output_num_signals_1:
        print(f"Signal count mismatch: first x/y num_signals ({input_num_signals_0}, {output_num_signals_0}), "
              f"last x/y num_signals ({input_num_signals_1}, {output_num_signals_1}). Removing last entries.")
        x_list = x_list[:-1]
        y_list = y_list[:-1]
    
    return x_list, y_list
    
    
# Manual batch splitting inside the training loop.
def rebatch_dict_of_lists(batch, batch_size, min_final_batch_size=10):
    x_list = batch['x']
    y_list = batch['y']
    # Each entry in x_ y_list is a list of tensors each one of shape :
    # (nr_features, time-window length).
    
    ## HINT:
    # The last tensor might have a smaller nr_signals. In fact, during the segmentation
    # in time windows, some signals might drop the last segment if the time length 
    # is not long enough. 

    # Check that the shape of the last element in the lists is coinsistent 
    # with the number of signals expected, e.g., the nr_signals found in the first batch
    x_list, y_list = compare_last_first_tensor_shapes(x_list, y_list)
        
    total = len(x_list)
    batches = []

    if total < batch_size:
        return [{'x': x_list, 'y': y_list}]
    
    i = 0
    while i < total and i + batch_size <= total:
        batches.append({
            'x': x_list[i:i+batch_size],
            'y': y_list[i:i+batch_size]
        })
        i += batch_size
    
    remaining = total - i
    if remaining > 0:
        if remaining < min_final_batch_size:
            last_batch = batches[-1]
            last_batch['x'].extend(x_list[i:])
            last_batch['y'].extend(y_list[i:])
        else:
            batches.append({
                'x': x_list[i:],
                'y': y_list[i:]
            })
            
    return batches

def train_model(
    model,
    device, 
    train_dataloader, 
    val_dataloader,
    batch_size,
    min_batch_size,
    num_epochs, 
    output_filename,
    criterion=nn.MSELoss(),
    optimiser=None):
    """
    Train the NeuralNetwork model.

    Parameters
    ----------
    model : NeuralNetwork
        The neural network model to train.
    device : torch.device
        The device to train the model on.
    train_dataloader : DataLoader
        DataLoader for the training data.
    val_dataloader : DataLoader
        DataLoader for the validation data.
    batch_size : int
        Number of batches in the training data.
    min_batch_size : int, optional
        Minimum size of the final batch, by default 10.
    num_epochs : int, optional
        Number of epochs to train the model.
    criterion : nn.Module, optional
        Loss function to use, by default nn.MSELoss().
    optimiser : torch.optim.Optimizer, optional
        Optimiser to use for training, by default None. If None, Adam optimiser is used
    """
    eval_loss_vs_epoch = []
    training_loss_vs_epoch = []
    for epoch in range(num_epochs):
        print(f"Epoch: {epoch}")
        model.train()
        running_loss = 0.0

        # Loop through base batches from the DataLoader
        for batch_idx, base_batch in enumerate(train_dataloader):
            if batch_size is not None and min_batch_size is not None:
                batches = rebatch_dict_of_lists(base_batch, batch_size, min_batch_size)
            else:
                batches = [base_batch]
            
            for batch in batches:
                x_batch = batch['x']
                y_batch = batch['y']

                x_tensor_list = []
                y_tensor_list = []
                for x_signals, y_signals in zip(x_batch, y_batch):
                    x_flat = torch.cat([sig.flatten() for sig in x_signals], dim=0)
                    y_flat = torch.cat([sig.flatten() for sig in y_signals], dim=0)
                    x_tensor_list.append(x_flat)
                    y_tensor_list.append(y_flat)

                    # Stack into batch tensors
                    x_tensor = torch.stack(x_tensor_list)  # [batch_size, input_dim]
                    y_tensor = torch.stack(y_tensor_list)  # [batch_size, target_dim]
                    
                    x_tensor = x_tensor.to(device)
                    y_tensor = y_tensor.to(device)
                    
                    optimiser.zero_grad()
                    outputs = model(x_tensor)

                    loss = criterion(outputs, y_tensor)

                    loss.backward()
                    optimiser.step()

                    running_loss += loss.item()
                    
        print(f"Train loss: {running_loss/len(train_dataloader):.2e}")
        training_loss_vs_epoch.append(running_loss/len(train_dataloader))
        
        # STARTING EVALUATION
        model.eval()
        running_loss = 0.0
        
        for batch_idx, base_batch in enumerate(val_dataloader):
            if batch_size is not None and min_batch_size is not None:
                batches = rebatch_dict_of_lists(base_batch, batch_size, min_batch_size)
            else:
                batches = [base_batch]
                
            for batch in batches:
                x_batch = batch['x']
                y_batch = batch['y']
                
                x_tensor_list = []
                y_tensor_list = []
                for x_signals, y_signals in zip(x_batch, y_batch):
                    x_flat = torch.cat([sig.flatten() for sig in x_signals], dim=0)
                    y_flat = torch.cat([sig.flatten() for sig in y_signals], dim=0)
                    x_tensor_list.append(x_flat)
                    y_tensor_list.append(y_flat)

                    # Stack into batch tensors
                    x_tensor = torch.stack(x_tensor_list)  # [batch_size, input_dim]
                    y_tensor = torch.stack(y_tensor_list)  # [batch_size, target_dim]
                    
                    x_tensor = x_tensor.to(device)
                    y_tensor = y_tensor.to(device)
                    
                    optimiser.zero_grad()
                    outputs = model(x_tensor)

                    loss = criterion(outputs, y_tensor)

                    running_loss += loss.item()
        print(f"Val loss: {running_loss/len(val_dataloader):.2e}")
        eval_loss_vs_epoch.append(running_loss/len(val_dataloader))
        
    return training_loss_vs_epoch, eval_loss_vs_epoch  

        
#================================================================
    ########## END of Model Definition and Training ##########
#================================================================

if __name__== "__main__":
    
    #============== INPUT and SET-UP SECTION ==============#
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pipeline_config_file",
        type=str,
        default="config_pipeline_my.json",
        help="Path to configuration file for the pipeline.")
    
    args = parser.parse_args()
    
    # Load configuration from JSON file
    config_file_path =  args.pipeline_config_file

    if not os.path.exists(config_file_path):
        raise FileNotFoundError(f"Configuration file {config_file_path} not found.")       
    
    # Initialize SETTINGS object
    SETTINGS = get_settings(config_file_path)
    
    # Load models from joblib files
    model_dictionary = load_models(SETTINGS.DATA.all_source_signal_list, 
                                   SETTINGS.LOCAL_PATHS.joblib_directory)
    
    #============== END INPUT and SET-UP SECTION ==============#

    
    
    ########## INITIALIZATION SECTION ##########
    # Train, val, and test shots
    train_shots, test_shots, val_shots = read_data_split_csv(SETTINGS.LOCAL_PATHS.data_split_csv_path)
    train_shots = train_shots[:SETTINGS.TRAINING.num_train_samples]  
    val_shots = val_shots[:SETTINGS.TRAINING.num_val_samples] # For testing purposes, limit the number of validation shots
    
    # Define transform pipelines for data and target signals
    transforms_set_for_data = ComposeTransforms(
        [
            ImputerTransform(model_dictionary,SETTINGS.LOCAL_PATHS.average_values_file_path),
            PCATransform(model_dictionary)
        ]
    )
    transforms_set_for_target = ComposeTransforms(
        [
            ImputerTransform(model_dictionary,SETTINGS.LOCAL_PATHS.average_values_file_path),
            PCATransform(model_dictionary)
        ]
    )
    
    
    # Make a map of transforms to apply at signal level
    signal_transform_map = { var: transforms_set_for_data for var in SETTINGS.DATA.data_names}
    signal_transform_map.update({ var: transforms_set_for_target for var in SETTINGS.DATA.target_names})

    #  Make a map of transforms to apply at shot level
    shot_level_transform = None # No shot level transform is applied
    
    # Initialize Datasets for training and validation
    dataset_for_training = initialize_dataset(
        local=SETTINGS.DATA.local,
        shots_list=train_shots,
        source_signal_list=SETTINGS.DATA.all_source_signal_list,
        signal_level_transform_map=signal_transform_map,
        shot_level_transform=shot_level_transform
    )
    
    dataset_for_validation = initialize_dataset(
        local=SETTINGS.DATA.local,
        shots_list=val_shots,
        source_signal_list=SETTINGS.DATA.all_source_signal_list,
        signal_level_transform_map=signal_transform_map,
        shot_level_transform=shot_level_transform
    )
    
    customised_collate_fn = MiniBatchCollateFn(
                data_names=SETTINGS.DATA.data_names,
                target_names=SETTINGS.DATA.target_names,
                time_window_sec=SETTINGS.TIME_SEGMENTATION.time_window_sec, 
                time_step=SETTINGS.TIME_SEGMENTATION.time_step, 
                offset=SETTINGS.TIME_SEGMENTATION.offset,
                )
    
    # Initialize DataLoaders for training and validation
    train_dataloader = initialize_dataloader(
        dataset=dataset_for_training,
        batch_size=SETTINGS.TRAINING.dataloader_batch_size,
        num_workers=SETTINGS.TRAINING.num_workers,
        shuffle=True,
        collate_fn = customised_collate_fn
    )
    
    val_dataloader = initialize_dataloader(
        dataset=dataset_for_validation,
        batch_size=SETTINGS.TRAINING.dataloader_batch_size,
        num_workers=SETTINGS.TRAINING.num_workers,
        shuffle=True,
        collate_fn = customised_collate_fn
    )


    # Determine device to train on
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # Model definition
    input_size, output_size = 0, 0
    # Try to get input and output size from the first batch of data
    # If it fails, it will raise a ValueError and we will print the error message
    try:
        input_size, output_size =  get_input_output_size(train_dataloader)
    except ValueError as e:
        print(f"Error getting input/output size: {e}")
        sys.exit(1)


    model = NeuralNetwork(
        input_size=input_size, 
        output_size = output_size,
        l1_size=SETTINGS.NEURALNET.l1_size,
        l2_size=SETTINGS.NEURALNET.l2_size
        ).to(device)
    
    criterion = nn.MSELoss() # Loss function to use, by default MSELoss.
    optimiser = torch.optim.Adam(model.parameters(), lr=SETTINGS.NEURALNET.lr)
    
    output_filename=SETTINGS.LOCAL_PATHS.data_output_directory + "NeuralNetwork.txt"
    train_loss, eval_loss = train_model(
        model,
        device,
        train_dataloader,
        val_dataloader,
        SETTINGS.TRAINING.train_batch_size,
        SETTINGS.TRAINING.min_batch_size,
        SETTINGS.TRAINING.num_epochs,
        output_filename,
        criterion,
        optimiser
        )
    

    print("Training completed successfully.")
    # Save the model
    torch.save({
        'model_state_dict': model.state_dict(),
        'hyperparameters': {
            'input_size': input_size,
            'output_size': output_size,
            'l1_size': SETTINGS.NEURALNET.l1_size,
            'l2_size': SETTINGS.NEURALNET.l2_size,
            'num_epochs': SETTINGS.TRAINING.num_epochs,
            'learning_rate': SETTINGS.NEURALNET.lr,
            'batch_size': SETTINGS.TRAINING.train_batch_size,
            'num_train_samples': SETTINGS.TRAINING.num_train_samples,
            'num_eval_samples': SETTINGS.TRAINING.num_val_samples,
            'dataloader_batch_size': SETTINGS.TRAINING.dataloader_batch_size,
            'num_workers': SETTINGS.TRAINING.num_workers,
            'device': str(device),
            'data_names': SETTINGS.DATA.data_names,
            'target_names': SETTINGS.DATA.target_names,
            "time_window_sec": SETTINGS.TIME_SEGMENTATION.time_window_sec, 
            "time_step": SETTINGS.TIME_SEGMENTATION.time_step, 
            "offset": SETTINGS.TIME_SEGMENTATION.offset,
            'local': SETTINGS.DATA.local,
            'train_loss': train_loss,
            'eval_loss': eval_loss
        }
    }, f"model{SETTINGS.NEURALNET.lr}.pth")
    
