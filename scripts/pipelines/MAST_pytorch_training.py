"""
Pipeline for MAST data forecasting 

-  DATA-SPLIT:
        f1) read_data_split_from_csv

- DATALOADER:
        C1) MASTDataset
        f1) load_models
        f2) create_dataloaders

- MODEL TRAINING:
        C1) NeuralNetwork
        f1) train_model
"""

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
sys.path.append(os.path.abspath(os.path.join(mother_dir , "MAST_tools")))
sys.path.append(mother_dir)


from signal_utils import MASTSignalManager  
from store_utils import MASTStorageManager

from transforms import (ImputerTransform, 
                        PCATransform, 
                        ComposeTransform)

from collate_functions import MiniBatchCollateFn
from  utils import (shuffle_shot_ids)


#================================================================
        ##########   DATA-SPLIT  ##########
#================================================================
def read_data_split_csv(csv_path="metadata/2025-05-12/data_splits.csv"):
    """Read the csv file containing the lists of shot IDs for 
    training, validation and testing.

    Parameters
    ----------
    csv_path : str, optional
        by default "metadata/2025-05-12/data_splits.csv"

    Returns
    -------
    Three lists containing shot IDs for training, validation and testing sets
    """

    df = pd.read_csv(csv_path)

    # Filter rows where the 'train' column is True
    shot_ids_for_train = df[df['train'] == True]['shot_id'].tolist() 
    shot_ids_for_test = df[df['test'] == True]['shot_id'].tolist() 
    shot_ids_for_val = df[df['val'] == True]['shot_id'].tolist() 

    return shot_ids_for_train, shot_ids_for_test, shot_ids_for_val
#================================================================
        ##########   END OF DATA-SPLIT  ##########
#================================================================



#================================================================
        ##########    DATALOADER  ##########
#================================================================

def load_models(data_names, data_dir):
    """Load the PCA and imputer models for the given data names.

    Parameters
    ----------
    data_names : list[str]
        List of data names to load models for.

    Returns
    -------
    dict
        Dictionary containing the loaded PCA and imputer models.
    """
    pca_models = {}
    imputer_models = {}

    for data_name in data_names:
        source_name, signal_name = data_name.split("-")
        pca_model_path = f"{data_dir}pca_{signal_name}.joblib"
        imputer_model_path = f"{data_dir}imputer_{signal_name}.joblib"
        pca_models[data_name] = joblib.load(pca_model_path)
        imputer_models[data_name] = joblib.load(imputer_model_path)

    return {"pca": pca_models, "imputer": imputer_models}


class MASTDataset(Dataset):
    """Dataset class for MAST data.

    Parameters
    ----------
    Dataset : torch.utils.data.Dataset
        See __init__ for details.
    """

    def __init__(
        self,local:bool,
        shots_list: list[int],
        data_names: list[str],
        target_names: list[str],
        transform_data=None,
        transform_target=None
        ):
        """ Initialize the MASTDataset.
        Parameters
        ----------
        local : bool
            If True, use local MAST database, otherwise use remote S3 bucket
        shots_list : list[int]
            List of shot IDs to load data for.
        data_names : list[str]
            List of data names to load (format: source_name-signal_name)
        target_names : list[str]
            List of target names to load (format: source_name-signal_name)
        transform_data : _type_, optional
            pipeline of transforms to apply to the data, by default None
        transform_target : _type_, optional
            pipeline of transforms to apply to the target, by default None
        """
        
        self.local = local
        self.shots_list = shots_list
        self.data_names = data_names
        self.target_names = target_names
        self.transform_data = transform_data
        self.transform_target = transform_target
        self.sig = MASTSignalManager()  
        self.store_manager =  MASTStorageManager()

    def __len__(self):
        return len(self.shots_list)

    def _collect_profiles(self, store, source_signal_names):
        profiles = []
        times = []
        for source_signal_name in source_signal_names:
            source_name, signal_name = source_signal_name.split("-")
            profile = self.sig.get_signal_profile(
                    data_origin=store,
                    source_name=source_name,
                    signal_name=signal_name,
                    verbose = False
                    )
            # profile might be None, this case will be handled by ImputerTransform
            profiles.append(profile)
            
        return profiles
        
        
    def _apply_transform(self, profile, source_signal_name, store, transform):
        """
        Apply transform to each profile for all signal in the list of signal_names
        """
        source_name, signal_name = source_signal_name.split("-")

        if  profile is not None :
            try:
                time, _ = self.sig.get_signal_times_and_time_type(
                signal_name,
                store,
                source_name
                )
                time = torch.from_numpy(time)
            except  AttributeError as e:
                time = torch.tensor([])
            try:
                vals = profile.values
            except  AttributeError:
                vals = np.array([])
        else:
            vals = np.array([])
            time = torch.tensor([])
            
        results = None
        if transform is not None:
            results = transform((vals, time, source_signal_name))
        
            if results is not None:
                vals, time, _ = results
                vals = torch.tensor(vals, dtype=torch.float32)
            else:
                vals = torch.tensor([])
        else:
            vals = torch.tensor(vals, dtype=torch.float32)
            
        return (vals, time)
        
        
    def __getitem__(self, idx):
        store = self.store_manager.make_shot_store(
            shot_info = {
                "shot_id":self.shots_list[idx], 
                "local":self.local 
                }
        )
 
        # Collect profiles for the data and target signals
        data_profiles = self._collect_profiles(store, self.data_names)
        target_profiles = self._collect_profiles(store, self.target_names)
      

        # Apply transforms and store structures of data and target signals
        target_entry = []
        for index, profile in enumerate(target_profiles):
            source_name, signal_name = self.target_names[index].split("-")
                
            y_vals, y_time = self._apply_transform(
                profile, 
                self.target_names[index],
                store,
                self.transform_target)           

            target_entry.append({   
                            "names":f"{source_name}-{signal_name}",
                            "time": y_time,
                            "values": y_vals
                        })
            
        data_entry = []
        for index, profile in enumerate(data_profiles):
            source_name, signal_name = self.data_names[index].split("-")         
            
            x_vals, x_time = self._apply_transform(
                profile,
                self.data_names[index],
                store,
                self.transform_data)
        
            data_entry.append({   
                            "names":f"{source_name}-{signal_name}",
                            "values": x_vals,
                            "time": x_time
                        } )
            
        # Data
        x = {
                "shot_id" : self.shots_list[idx],
                "source_name-signal_name": data_entry
            }

        # Target
        y = {
                "shot_id" : self.shots_list[idx], 
                "source_name-signal_name": target_entry
            }
        
        return x, y
#================================================================
    
    
# If Segmentertransform is used, it expands each sample into multiple sub-samples.
# When __getitem__ returns those multiple sub-units (as a batch-like structure), 
# the DataLoader ends up creating batches of batches, which may blow up memory and batch size.
# For instance, if the batch size is 32 and each sample is expanded into 10 sub-samples,
# the DataLoader will try to create a batch of 320 samples, which may not fit into memory.
# In this case use batch_size=1 in the DataLoader to avoid this issue then customise the collate function
# to handle the batching of sub-samples.
def create_dataloaders(
    time_window_sec,
    time_step,
    offset,
    train_shots, 
    val_shots,
    test_shots,
    data_names,
    target_names,
    batch_size, 
    num_workers, 
    local,
    transform_data,
    transform_target
    ):
    """Create DataLoaders for training, validation and testing datasets.

    Parameters
    ----------
    time_window_sec : float
        Length of the time window in seconds to segment the x and y values.
    time_step : float
        Step in seconds to move the time window.
    offset : float
        Offset in seconds to start the time window from the end of the signal.
    train_shots : list[int]
        list of shot IDs for training
    val_shots : list[int]
        list of shot IDs for validation
    test_shots : list[int]
        list of shot IDs for testing
    data_names : list[str]
        list of data names to load (format: source_name-signal_name)
    target_names : list[str]
        list of target names to load (format: source_name-signal_name)
    batch_size : int
        Size of the batch
    num_workers : int
        Number of workers for DataLoader
    local : bool
        If True, use local MAST database, otherwise use remote S3 bucket
    transform_data : 
        Dictionary containing the transforms for train, val and test to apply to the data
    transform_target : 
        Dictionary containing the transforms for train, val and test to apply to the target
    Returns
    -------
    DataLoader
    """
    list_data_splits =  ["train","val", "test"]
    dict_shot_ids = {"train": train_shots, "val": val_shots, "test": test_shots}
    dict_datasets = {}
    dict_dataloaders = {}

    for data_split in list_data_splits: 
        dict_datasets[data_split] = MASTDataset(
            local,
            dict_shot_ids[data_split], 
            data_names=data_names, 
            target_names=target_names,
            transform_data=transform_data[data_split] if transform_data else None,
            transform_target=transform_target[data_split] if transform_target else None
            )
        
        try:
            customised_collate_fn = MiniBatchCollateFn(
                time_window_sec=time_window_sec, 
                time_step=time_step, 
                offset=offset
                )
        except ValueError as e:
            raise ValueError(f"Customised collate function failed to initialize: {e}")

        dict_dataloaders[data_split] = DataLoader(
            dict_datasets[data_split],
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=True,
            collate_fn=customised_collate_fn
        )
            
    return dict_dataloaders["train"],  dict_dataloaders["val"],  dict_dataloaders["test"]

#================================================================
        ##########    END OF DATALOADER  ##########
#================================================================


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
    with the number of signals expected, e.g., the nr_signals found in the first batch

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
    with open(config_file_path, 'r') as f:
        config = json.load(f)
    
    # Access mode
    local = config["local"] 
     
    # list of data and target names for input and output signals
    data_names = config["input"]["data_names"]
    target_names = config["input"]["target_names"]
    time_window_sec = config["input"]["time_window_sec"] # This is the length of the time window in sec. for segmentation of data signals.
    time_step = config["input"]["time_step"] # This is the time step in sec. Time window is moved by this amount backwards in time.
    offset = config["input"]["offset"] # This is the target offset in sec. Signal is predicted in this time window
    
    # Training parameters
    num_epochs = config["training"]["num_epochs"]
    dataloader_batch_size = config["training"]["dataloader_batch_size"]
    train_batch_size = config["training"]["training_batch_size"]
    min_batch_size = config["training"]["min_batch_size"]
    num_workers = config["training"]["num_workers"]
    
    # For testing porpuses reduce the sample size of training and validation sets
    num_train = config["training"]["num_train_samples"] # For testing purposes, limit the number of training shots
    val_shots = config["training"]["num_eval_samples"] # For testing purposes, limit the number of validation shots
    
    
    # Learning rate and model sizes
    lr = config["nn_model"]["lr"]
    l1_size = config["nn_model"]["l1_size"]
    l2_size = config["nn_model"]["l2_size"]
    
    # Create sets of shot IDs for training, validation and testing
    train_shots, test_shots, val_shots = read_data_split_csv(config["paths"]["data_split_csv_path"])
    train_shots = train_shots[:num_train]  
    val_shots = val_shots[:num_train] # For testing purposes, limit the number of validation shots
    train_shots = shuffle_shot_ids(train_shots)
    val_shots = shuffle_shot_ids(val_shots)
    
    # Load models from joblib files
    model_dictionary = load_models(data_names, data_dir=config["paths"]["joblib_directory"])
    composed_transforms = ComposeTransform(
        [
            ImputerTransform(model_dictionary,config["paths"]["average_values_file_path"]),
            PCATransform(model_dictionary)
        ]
    )
    
    transforms_dictionary_data = {
        "train": composed_transforms,
        "val": composed_transforms,
        "test": composed_transforms
    }
    transforms_dictionary_target = {
        "train": composed_transforms,
        "val": composed_transforms,
        "test": composed_transforms
    }
    
    train_dataloader, val_dataloader, test_dataloader = create_dataloaders(
        time_window_sec,  
        time_step,
        offset,
        train_shots, 
        val_shots,
        test_shots,
        data_names,
        target_names,
        dataloader_batch_size,
        num_workers,
        local = local,
        transform_data = transforms_dictionary_data,
        transform_target = transforms_dictionary_target
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
        l1_size=l1_size,
        l2_size=l2_size
        ).to(device)
    criterion = nn.MSELoss() # Loss function to use, by default MSELoss.
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)
    
    output_filename=f"nn3Layer{('_'.join(data_names))}_Ntrain{num_train}_Nepochs{num_epochs}_lr{lr}.txt"
    train_loss, eval_loss = train_model(
        model,
        device,
        train_dataloader,
        val_dataloader,
        train_batch_size,
        min_batch_size,
        num_epochs,
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
            'l1_size': l1_size,
            'l2_size': l2_size,
            'num_epochs': num_epochs,
            'learning_rate': lr,
            'batch_size': train_batch_size,
            'num_train_samples': num_train,
            'num_eval_samples': val_shots,
            'dataloader_batch_size': dataloader_batch_size,
            'num_workers': num_workers,
            'device': str(device),
            'data_names': data_names,
            'target_names': target_names,
            'time_window_sec': time_window_sec,
            'time_step': time_step,
            'offset': offset,
            'local': local,
            'train_loss': train_loss,
            'eval_loss': eval_loss
        }
    }, f"model{lr}.pth")
    
