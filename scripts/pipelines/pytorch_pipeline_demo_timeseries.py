"""
Demonstration of a time series forecasting data loading, preprocessing
and PyTorch model fitting pipeline with hyperparameter optimization.

The following steps are performed:

1. Synthetic Time Series Data Generation:
   - Generate synthetic time series dataset with multiple signals per
     example
   - Each example has 3 input signals and 2 target signals
   - Variable length time series between examples (90-120 timesteps)
   - Split into training (100), validation (20), and test (20) examples
   - Save the datasets as .npz files

2. Data Preprocessing:
   - Fit a PCA model to reduce 3D input signals to 2D at each time slice
   - Fit a StandardScaler to standardize the PCA-transformed features

3. Dataset and DataLoader Creation:
   - Custom PyTorch Dataset class to load time series data
   - Returns full time series as dictionaries
   - PCA transform applied to each time slice
   - StandardScaler applied to PCA-transformed features
   - Chunking functionality to create input/target sequences

4. Neural Network Model Definition:
   - Define an LSTM for time series forecasting

5. Hyperparameter Optimization with Optuna:
   - Optimize learning rate, optimizer choice, batch size, hidden
     dimensions, num layers
   - Early stopping based on validation loss
   - TensorBoard logging for training visualization

6. Final Model Evaluation on Test Set
"""

import argparse
import datetime
import logging
import os
import numpy as np
import torch
import joblib
import optuna
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from tensordict import TensorDict
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Set up logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pytorch_pipeline_demo_timeseries.log'),
        # logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# =====================
# Data Generation
# =====================


def generate_synthetic_timeseries_data():
    """
    Generate synthetic time series data with multiple signals per example.
    
    Each example contains:
    - 3 input signals
    - 2 target signals
    - Variable length between 90-120 timesteps
    """
    np.random.seed(42)
    
    # Define dataset splits
    n_train, n_val, n_test = 100, 20, 20
    splits = [("train", n_train), ("val", n_val), ("test", n_test)]
    
    os.makedirs("data", exist_ok=True)
    
    for split_name, n_examples in splits:
        split_dir = os.path.join("data", split_name)
        os.makedirs(split_dir, exist_ok=True)
        
        for i in range(n_examples):
            # Random length for this example
            length = np.random.randint(90, 121)
            
            # Generate time series data
            t = np.linspace(0, 4*np.pi, length)
            
            # Create 3 input signals with different characteristics
            input_1 = np.sin(t) + 0.1 * np.random.randn(length)
            input_2 = np.cos(t + np.pi/4) + 0.1 * np.random.randn(length)
            input_3 = np.sin(2*t) * np.cos(t/2) + 0.1 * np.random.randn(length)
            
            # Create 2 target signals that depend on inputs
            target_1 = 0.5 * input_1 + 0.3 * input_2 + 0.05 * \
                       np.random.randn(length)
            target_2 = 0.4 * input_2 + 0.6 * input_3 + 0.05 * \
                       np.random.randn(length)
            
            # Add some lag relationship for forecasting
            if length > 5:
                target_1[5:] += 0.2 * input_1[:-5]  # target depends on past input
                target_2[5:] += 0.2 * input_3[:-5]
            
            # Save as individual files
            data = {
                'input_1': input_1,
                'input_2': input_2,
                'input_3': input_3,
                'target_1': target_1,
                'target_2': target_2,
                'length': length
            }
            
            np.savez(os.path.join(split_dir, f"example_{i:03d}.npz"), **data)
    
    logger.info("Time series data saved in data/ subdirectories")


# =====================
# Preprocessing
# =====================


def fit_and_save_pca(
        train_data_dir,
        n_components=2,
        model_path="pca_model.joblib"
    ):
    """
    Fit PCA to time slices of the 3 input signals to reduce from 3D to
    2D.
    
    Parameters
    ----------
    train_data_dir : str
        Directory where the train data files are located.
    n_components : int
        Number of PCA components (should be 2 for this example).
    model_path : str
        File path to save the fitted PCA model.
    """
    # Gather all time slices from training data
    all_time_slices = []
    
    train_files = sorted(
        [f for f in os.listdir(train_data_dir) if f.endswith('.npz')]
    )
    
    for file in train_files:
        data = np.load(os.path.join(train_data_dir, file))
        
        # For each time step, create a 3D vector
        # [input_1[t], input_2[t], input_3[t]]
        length = len(data['input_1'])
        for t in range(length):
            time_slice = np.array([
                data['input_1'][t],
                data['input_2'][t],
                data['input_3'][t]
            ])
            all_time_slices.append(time_slice)
    
    # Convert to array: (n_time_slices, 3)
    all_time_slices = np.array(all_time_slices)
    
    # Fit PCA
    pca = PCA(n_components=n_components)
    pca.fit(all_time_slices)
    
    joblib.dump(pca, model_path)
    logger.info(f"PCA model saved to {model_path}")
    logger.info(f"PCA explained variance ratio: {pca.explained_variance_ratio_}")


def fit_and_save_scaler(
        train_data_dir,
        pca_model_path="pca_model.joblib", 
        scaler_model_path="scaler_model.joblib"
    ):
    """
    Fit StandardScaler to PCA-transformed training data.
    
    Parameters
    ----------
    train_data_dir : str
        Directory where the train data files are located.
    pca_model_path : str
        Path to fitted PCA model.
    scaler_model_path : str
        File path to save the fitted StandardScaler model.
    """
    # Load PCA model
    pca_model = joblib.load(pca_model_path)
    
    # Gather all PCA-transformed time slices from training data
    all_pca_features = []
    
    train_files = sorted(
        [f for f in os.listdir(train_data_dir) if f.endswith('.npz')]
    )
    
    for file in train_files:
        data = np.load(os.path.join(train_data_dir, file))
        
        # For each time step, create a 3D vector and apply PCA
        length = len(data['input_1'])
        for t in range(length):
            time_slice = np.array([
                data['input_1'][t],
                data['input_2'][t],
                data['input_3'][t]
            ]).reshape(1, -1)
            
            # Apply PCA transform
            pca_features = pca_model.transform(time_slice).squeeze()
            all_pca_features.append(pca_features)
    
    # Convert to array: (n_time_slices, 2)
    all_pca_features = np.array(all_pca_features)
    
    # Fit StandardScaler
    scaler = StandardScaler()
    scaler.fit(all_pca_features)
    
    joblib.dump(scaler, scaler_model_path)
    logger.info(f"StandardScaler model saved to {scaler_model_path}")
    logger.debug(f"Scaler mean: {scaler.mean_}")
    logger.debug(f"Scaler scale: {scaler.scale_}")


# =====================
# Dataset and Transforms
# =====================


class PCATransform(object):
    """
    Apply pre-fitted PCA to transform 3D input signals to 2D at each
    time slice.
    
    Parameters
    ----------
    pca_model_path : str
        Path to fitted PCA model joblib file.
    """
    
    def __init__(self, pca_model_path):
        self.pca_model = joblib.load(pca_model_path)
    
    def __call__(self, sample):
        x, y = sample
        length = len(x['input_1'])
        
        # Create array for PCA-transformed inputs
        transformed_inputs = np.zeros((length, 2))  # 2 PCA components
        
        for t in range(length):
            # Create time slice vector
            time_slice = np.array([
                x['input_1'][t].item(),
                x['input_2'][t].item(), 
                x['input_3'][t].item()
            ]).reshape(1, -1)
            
            # Apply PCA transform
            transformed_slice = self.pca_model.transform(time_slice).squeeze()
            transformed_inputs[t] = transformed_slice
        
        # Replace x dictionary with PCA-transformed data
        x_transformed = {
            'pca_1': torch.from_numpy(transformed_inputs[:, 0]).float(),
            'pca_2': torch.from_numpy(transformed_inputs[:, 1]).float()
        }
        
        return x_transformed, y


class StandardScalerTransform(object):
    """
    Apply pre-fitted StandardScaler to PCA-transformed input signals.
    
    Parameters
    ----------
    scaler_model_path : str
        Path to fitted StandardScaler model joblib file.
    """
    
    def __init__(self, scaler_model_path):
        self.scaler_model = joblib.load(scaler_model_path)
    
    def __call__(self, sample):
        x, y = sample
        
        # Stack PCA components for scaling
        pca_features = np.column_stack([
            x['pca_1'].numpy(),
            x['pca_2'].numpy()
        ])
        
        # Apply StandardScaler
        scaled_features = self.scaler_model.transform(pca_features)
        
        # Replace x dictionary with scaled data
        x_scaled = {
            'pca_1': torch.from_numpy(scaled_features[:, 0]).float(),
            'pca_2': torch.from_numpy(scaled_features[:, 1]).float()
        }
        
        return x_scaled, y


class TimeSeriesDataset(Dataset):
    """
    Custom Dataset for loading time series data stored in .npz files.
    
    Returns full time series as dictionaries for x and y.

    Parameters
    ----------
    data_dir : str
       Path to directory containing .npz files.
    transform : callable, optional
      Optional transform to apply to the data.
    """
    
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        
        self.data_files = sorted([
            f for f in os.listdir(self.data_dir) 
            if f.endswith('.npz')
        ])
    
    def __len__(self):
        return len(self.data_files)
    
    def __getitem__(self, idx):
        file_path = os.path.join(self.data_dir, self.data_files[idx])
        data = np.load(file_path)
        
        x = {
            'input_1': torch.from_numpy(data['input_1']).float(),
            'input_2': torch.from_numpy(data['input_2']).float(), 
            'input_3': torch.from_numpy(data['input_3']).float()
        }
        
        y = {
            'target_1': torch.from_numpy(data['target_1']).float(),
            'target_2': torch.from_numpy(data['target_2']).float()
        }
        
        if self.transform:
            x, y = self.transform((x, y))
        
        return x, y, file_path


def chunk_time_series(x_dict, y_dict, file_path, input_length=5, target_length=3):
    """
    Chunk time series into input/target sequences for forecasting.
    
    Parameters
    ----------
    x_dict : dict
        Dictionary with input time series (after PCA and scaling:
        'pca_1', 'pca_2')
    y_dict : dict  
        Dictionary with target time series
    file_path : str
        Path to the raw input data file.
    input_length : int
        Length of input sequences
    target_length : int
        Length of target sequences to predict
        
    Returns
    -------
    list of TensorDict
        List of chunked sequences as TensorDicts
    """
    chunks = []
    
    # Get series length (assuming all series in an example have same length)
    series_length = len(x_dict['pca_1'])
    
    # Create chunks
    for i in range(series_length - input_length - target_length + 1):
        # Input chunk: 5 preceding values of 2 PCA components (scaled)
        input_chunk = TensorDict({
            'pca_1': x_dict['pca_1'][i:i+input_length],
            'pca_2': x_dict['pca_2'][i:i+input_length]
        }, batch_size=[])
        
        # Target chunk: 3 following values of 2 target signals
        target_start = i + input_length
        target_chunk = TensorDict({
            'target_1': y_dict['target_1'][target_start:target_start+target_length],
            'target_2': y_dict['target_2'][target_start:target_start+target_length]
        }, batch_size=[])
        
        chunk = TensorDict({
            'x': input_chunk,
            'y': target_chunk,
            'file_path': file_path
        }, batch_size=[])
        
        chunks.append(chunk)
    
    return chunks


def collate_chunks(batch):
    """
    Custom collate function to combine chunks into TensorDict batches.

    Parameters
    ----------
    batch : list
        List of tuples where each tuple contains (x_dict, y_dict)
        representing input and target dictionaries for a single time
        series example.

    Returns
    -------
    TensorDict
        Batched TensorDict containing stacked chunks from all examples
        in the batch. Each chunk has 'x' (input sequences) and 'y'
        (target sequences). If no chunks are available, returns empty
        TensorDict with batch_size [0].
    """
    # Flatten all chunks from all examples in the batch
    all_chunks = []
    for x_dict, y_dict, file_path in batch:
        chunks = chunk_time_series(x_dict, y_dict, file_path)
        all_chunks.extend(chunks)
    
    if not all_chunks:
        return TensorDict({}, batch_size=[0])
    
    # Stack chunks into batched TensorDict
    return torch.stack(all_chunks, dim=0)


def create_dataloaders(batch_size=32, num_workers=2, transform=None):
    """
    Create data loaders for train, validation, and test datasets.

    Parameters
    ----------
    batch_size : int, optional
        Number of samples per batch. Default is 32.
    num_workers : int, optional
        Number of subprocesses to use for data loading. Default is 2.
    transform : callable, optional
        Optional transform to be applied to the data.

    Returns
    -------
    tuple
        Tuple containing (train_dataloader, val_dataloader,
        test_dataloader) where each element is a
        torch.utils.data.DataLoader for the respective dataset split.
    """
    train_data_dir = os.path.join("data", "train")
    val_data_dir = os.path.join("data", "val") 
    test_data_dir = os.path.join("data", "test")
    
    train_dataset = TimeSeriesDataset(train_data_dir, transform=transform)
    val_dataset = TimeSeriesDataset(val_data_dir, transform=transform)
    test_dataset = TimeSeriesDataset(test_data_dir, transform=transform)
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        collate_fn=collate_chunks
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size, 
        num_workers=num_workers,
        shuffle=False,
        collate_fn=collate_chunks
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        num_workers=num_workers, 
        shuffle=False,
        collate_fn=collate_chunks
    )
    
    return train_dataloader, val_dataloader, test_dataloader


# =====================
# Model Definition and Training
# =====================


class TimeSeriesForecastingModel(nn.Module):
    """
    Neural network for time series forecasting.
    
    Takes 5 timesteps of 2 PCA-transformed and standardized input signals 
    and predicts 3 timesteps of 2 target signals.

    Parameters
    ----------
    input_dim : int, optional
        Number of input features. Default is 2.
    hidden_dim : int, optional
       Number of hidden units. Default is 64.
    num_layers : int, optional
        Number of LSTM layers. Default is 1.
    """
    
    def __init__(self, input_dim=2, hidden_dim=64, num_layers=1):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Input: (batch_size, 5 timesteps, 2 PCA components)
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        self.lstm = nn.LSTM(
            hidden_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )
        
        # Output: (batch_size, 3 timesteps, 2 signals)  
        self.output_projection = nn.Linear(hidden_dim, 2)
        self.output_length = 3
    
    def forward(self, x_tensordict):
        # Extract PCA-transformed and scaled inputs from TensorDict
        # x_tensordict['x'] has keys 'pca_1', 'pca_2'
        inputs = torch.stack([
            x_tensordict['x']['pca_1'],
            x_tensordict['x']['pca_2']
        ], dim=-1)  # (batch_size, 5, 2)
        
        # Project inputs
        x = self.input_projection(inputs)  # (batch_size, 5, hidden_dim)
        
        # LSTM processing
        lstm_out, (h_n, c_n) = self.lstm(x)  # (batch_size, 5, hidden_dim)
        
        # Use final hidden state to generate predictions
        final_hidden = h_n[-1]  # (batch_size, hidden_dim)
        
        # Generate 3 timesteps of predictions
        predictions = []
        hidden = final_hidden
        
        for _ in range(self.output_length):
            pred = self.output_projection(hidden)  # (batch_size, 2)
            predictions.append(pred)
            # Simple recurrence for next timestep
            hidden = hidden + 0.1 * pred.sum(dim=-1, keepdim=True) * \
                     torch.randn_like(hidden)
        
        predictions = torch.stack(predictions, dim=1)  # (batch_size, 3, 2)
        
        return TensorDict({
            'target_1': predictions[:, :, 0],  # (batch_size, 3)
            'target_2': predictions[:, :, 1]   # (batch_size, 3)
        }, batch_size=predictions.shape[0])


class EarlyStopping:
    """Early stopping utility class.
    
    Parameters
    ----------
    patience : int, optional
        Number of epochs to wait before considering stopping. Default is
        7.
    min_delta : float, optional
        Minimum delta to consider as improvement. Default is 0.
    restore_best_weights : bool, optional
        Restore the best weights found during training. Default is True.
    """
    
    def __init__(self, patience=7, min_delta=0.0, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights

        self.best_loss = float('inf')
        self.counter = 0
        self.best_weights = None
    
    def __call__(self, val_loss, model):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            if self.restore_best_weights:
                self.best_weights = model.state_dict().copy()
        else:
            self.counter += 1
        
        if self.counter >= self.patience:
            if self.restore_best_weights and self.best_weights is not None:
                model.load_state_dict(self.best_weights)
            return True
        return False


def train_model_with_params(
        device,
        train_dataloader,
        val_dataloader,
        hyperparams,
        trial_num=None,
        writer=None,
        max_epochs=100
    ):
    """
    Train the time series forecasting model with given hyperparameters.
    
    Returns the best validation loss achieved.

    Parameters
    ----------
    device : torch.device
        Device to run the model on (CPU, CUDA, or MPS).
    train_dataloader : torch.utils.data.DataLoader
        DataLoader for training data.
    val_dataloader : torch.utils.data.DataLoader
        DataLoader for validation data.
    hyperparams : dict
        Dictionary containing hyperparameters with keys:
        'lr' (float), 'optimizer' (str), 'hidden_dim' (int),
        'num_layers' (int).
    trial_num : int, optional
        Trial number for logging purposes. Default is None.
    writer : torch.utils.tensorboard.SummaryWriter, optional
        TensorBoard writer for logging. Default is None.
    max_epochs : int, optional
        Maximum number of training epochs. Default is 100.

    Returns
    -------
    tuple
        Tuple containing (best_val_loss, model) where best_val_loss is
        float and model is the trained TimeSeriesForecastingModel.
    """
    # Create trial-specific logger
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    trial_suffix = f"_trial_{trial_num}" if trial_num is not None else ""
    log_filename = f"training_{timestamp}{trial_suffix}.log"
    
    trial_logger = logging.getLogger(f"training{trial_suffix}")
    trial_logger.setLevel(logging.INFO)
    
    # Remove any existing handlers to avoid duplication
    for handler in trial_logger.handlers[:]:
        trial_logger.removeHandler(handler)
    
    # Add file handler for this specific training run
    file_handler = logging.FileHandler(log_filename)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    file_handler.setFormatter(formatter)
    trial_logger.addHandler(file_handler)
    
    trial_logger.info(f"Starting training with hyperparameters: {hyperparams}")
    
    # Extract hyperparameters
    lr = hyperparams['lr']
    optimizer_name = hyperparams['optimizer']
    hidden_dim = hyperparams['hidden_dim']
    num_layers = hyperparams['num_layers']
    
    # Create model
    model = TimeSeriesForecastingModel(
        input_dim=2, 
        hidden_dim=hidden_dim, 
        num_layers=num_layers
    ).to(device)
    
    trial_logger.info(
        f"Model created with hidden_dim={hidden_dim}, num_layers={num_layers}"
    )
    
    # Create optimizer
    if optimizer_name == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    elif optimizer_name == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    elif optimizer_name == 'rmsprop':
        optimizer = torch.optim.RMSprop(model.parameters(), lr=lr)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")
    
    trial_logger.info(f"Optimizer: {optimizer_name}, Learning rate: {lr}")
    
    criterion = nn.MSELoss()
    early_stopping = EarlyStopping(patience=10, min_delta=1e-6)
    
    best_val_loss = float('inf')
    
    for epoch in range(max_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_batches = 0
        
        for batch_tensordict in train_dataloader:
            if batch_tensordict.batch_size[0] == 0:
                continue
                
            batch_tensordict = batch_tensordict.to(device)
            
            # Forward pass
            predictions = model(batch_tensordict)
            
            # Calculate loss
            targets = batch_tensordict['y']
            loss = (criterion(predictions['target_1'], targets['target_1']) + 
                   criterion(predictions['target_2'], targets['target_2']))
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_batches += 1
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_batches = 0
        
        with torch.no_grad():
            for batch_tensordict in val_dataloader:
                if batch_tensordict.batch_size[0] == 0:
                    continue
                    
                batch_tensordict = batch_tensordict.to(device)
                
                predictions = model(batch_tensordict)
                targets = batch_tensordict['y']
                
                loss = (criterion(predictions['target_1'], targets['target_1']) + 
                       criterion(predictions['target_2'], targets['target_2']))
                
                val_loss += loss.item()
                val_batches += 1
        
        # Calculate average losses
        avg_train_loss = train_loss / max(train_batches, 1)
        avg_val_loss = val_loss / max(val_batches, 1)
        
        # Log training progress
        trial_logger.info(
            f"Epoch {epoch}: " +
            f"Train Loss: {avg_train_loss:.6f}, " +
            f"Val Loss: {avg_val_loss:.6f}"
        )
        
        # Log to tensorboard if writer is provided
        if writer is not None:
            tag_prefix = f"trial_{trial_num}/" if trial_num is not None else ""
            writer.add_scalar(f"{tag_prefix}train_loss", avg_train_loss, epoch)
            writer.add_scalar(f"{tag_prefix}val_loss", avg_val_loss, epoch)
            writer.add_scalar(f"{tag_prefix}learning_rate", lr, epoch)
        
        # Update best validation loss
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            trial_logger.info(
                f"New best validation loss: {best_val_loss:.6f} at epoch {epoch}"
            )
        
        # Early stopping
        if early_stopping(avg_val_loss, model):
            trial_logger.info(f"Early stopping at epoch {epoch}")
            break
    
    trial_logger.info(
        f"Training completed. Best validation loss: {best_val_loss:.6f}"
    )
    
    # Clean up logger handlers
    for handler in trial_logger.handlers[:]:
        handler.close()
        trial_logger.removeHandler(handler)
    
    return best_val_loss, model


def evaluate_model(model, dataloader, device):
    """Evaluate model on a dataset and return average loss.
    
    Parameters
    ----------
    model : torch.nn.Module
        The trained model to evaluate.
    dataloader : torch.utils.data.DataLoader
        DataLoader containing the dataset to evaluate on.
    device : torch.device
        Device to run the evaluation on (CPU, CUDA, or MPS).

    Returns
    -------
    float
        Average loss across all batches in the dataset.
    """
    model.eval()
    criterion = nn.MSELoss()
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch_tensordict in dataloader:
            if batch_tensordict.batch_size[0] == 0:
                continue
                
            batch_tensordict = batch_tensordict.to(device)
            
            predictions = model(batch_tensordict)
            targets = batch_tensordict['y']
            
            loss = (criterion(predictions['target_1'], targets['target_1']) + 
                   criterion(predictions['target_2'], targets['target_2']))
            
            total_loss += loss.item()
            num_batches += 1
    
    return total_loss / max(num_batches, 1)


# =====================
# Hyperparameter Optimization
# =====================


def objective(trial, device, composed_transform, num_workers=2):
    """
    Optuna objective function for hyperparameter optimization.

    Parameters
    ----------
    trial : optuna.trial.Trial
        Optuna trial object for suggesting hyperparameters.
    device : torch.device
        Device to run the model on (CPU, CUDA, or MPS).
    composed_transform : torchvision.transforms.Compose
        Composed transform pipeline for data preprocessing.
    num_workers : int, optional
        Number of workers for data loading. Default is 2.

    Returns
    -------
    float
        Best validation loss achieved during training with the suggested
        hyperparameters.
    """
    # Suggest hyperparameters
    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    optimizer_name = trial.suggest_categorical(
        "optimizer",
        ["adam", "sgd", "rmsprop"]
    )
    batch_size = trial.suggest_categorical("batch_size", [8, 16, 32, 64])
    hidden_dim = trial.suggest_categorical("hidden_dim", [32, 64, 128, 256])
    num_layers = trial.suggest_int("num_layers", 1, 3)
    
    # Create data loaders with suggested batch size
    train_dataloader, val_dataloader, _ = create_dataloaders(
        batch_size=batch_size,
        num_workers=num_workers,
        transform=composed_transform
    )
    
    # Prepare hyperparameters
    hyperparams = {
        'lr': lr,
        'optimizer': optimizer_name,
        'hidden_dim': hidden_dim,
        'num_layers': num_layers
    }
    
    # Create tensorboard writer for this trial
    log_dir = f"runs/trial_{trial.number}"
    writer = SummaryWriter(log_dir)
    
    try:
        # Train model
        best_val_loss, _ = train_model_with_params(
            device=device,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            hyperparams=hyperparams,
            trial_num=trial.number,
            writer=writer,
            max_epochs=50
        )
        
        writer.close()
        return best_val_loss
        
    except Exception as e:
        writer.close()
        logger.error(f"Trial {trial.number} failed with error: {e}")
        return float('inf')


def run_hyperparameter_optimization(
        device,
        composed_transform,
        n_trials=20,
        n_workers=2
    ):
    """
    Run hyperparameter optimization using Optuna.

    Parameters
    ----------
    device : torch.device
        Device to run the model on (CPU, CUDA, or MPS).
    composed_transform : torchvision.transforms.Compose
        Composed transform pipeline for data preprocessing.
    n_trials : int, optional
        Number of trials to run for hyperparameter optimization. Default
        is 20.
    n_workers : int, optional
        Number of workers to use during data loading. Default is 2.

    Returns
    -------
    dict
        Dictionary containing the best hyperparameters found during
        optimization.
    """
    # Create optuna study
    study = optuna.create_study(direction="minimize")
    
    # Optimize
    study.optimize(
        lambda trial: objective(
            trial,
            device,
            composed_transform,
            n_workers
        ),
        n_trials=n_trials
    )
    
    # Log results
    logger.info("Best trial:")
    trial = study.best_trial
    logger.info(f"  Value: {trial.value}")
    logger.info("  Params: ")
    for key, value in trial.params.items():
        logger.info(f"    {key}: {value}")

    plot_optimization_history(study)
    
    return study.best_params


def train_final_model(device, best_params, composed_transform, num_workers=2):
    """
    Train the final model with best hyperparameters and evaluate on test
    set.

    Parameters
    ----------
    device : torch.device
        Device to run the model on (CPU, CUDA, or MPS).
    best_params : dict
        Dictionary containing the best hyperparameters found during
        optimization. Should contain keys: 'lr', 'optimizer',
        'batch_size', 'hidden_dim', 'num_layers'.
    composed_transform : torchvision.transforms.Compose
        Composed transform pipeline for data preprocessing.
    num_workers : int, optional
        Number of workers to use in data loading. Default is 2.

    Returns
    -------
    tuple
        Tuple containing (final_model, test_loss) where final_model is
        the trained TimeSeriesForecastingModel and test_loss is the
        float evaluation loss on the test set.
    """
    logger.info("Training final model with best hyperparameters")
    
    # Create data loaders with best batch size
    train_dataloader, val_dataloader, test_dataloader = create_dataloaders(
        batch_size=best_params['batch_size'],
        num_workers=num_workers,
        transform=composed_transform
    )
    
    # Create tensorboard writer for final training
    writer = SummaryWriter("runs/final_model")
    
    # Train final model
    best_val_loss, final_model = train_model_with_params(
        device=device,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        hyperparams=best_params,
        trial_num=None,
        writer=writer,
        max_epochs=100
    )
    
    # Evaluate on test set
    test_loss = evaluate_model(final_model, test_dataloader, device)
    
    logger.info(f"Final validation loss: {best_val_loss:.6f}")
    logger.info(f"Test loss: {test_loss:.6f}")
    
    writer.close()
    
    # Save final model
    torch.save({
        'model_state_dict': final_model.state_dict(),
        'hyperparams': best_params,
        'val_loss': best_val_loss,
        'test_loss': test_loss
    }, 'best_model.pth')
    
    return final_model, test_loss


# =====================
# Plotting
# =====================

def plot_optimization_history(study):
    """Plot optimization history.
    
    Parameters
    ----------
    study : optuna.study.Study
        Optuna study object.
    """
    logger.info(f"Plotting optimization history")
    ax = optuna.visualization.matplotlib.plot_optimization_history(study)
    ax.set_title("Optimization History")
    fig = ax.get_figure()
    figure_filename = "optuna_results.png"
    fig.savefig(figure_filename, dpi=300, bbox_inches='tight')
    logger.info(f"Figure saved as {figure_filename}")


def create_prediction_gif(
    model, 
    dataset, 
    device,
    example_idx=0, 
    filename="prediction_evolution.gif", 
    input_length=5, 
    target_length=3,
    fps=5
):
    """
    Create an animated GIF showing how inputs, targets, and predictions 
    evolve through time for a single example.
    
    Parameters
    ----------
    model : TimeSeriesForecastingModel
        Trained model for making predictions
    dataset : TimeSeriesDataset
        Dataset containing the examples
    device : torch.device
        Device to run the model on
    example_idx : int, optional
        Index of the example to visualize. Default is 0.
    filename : str, optional
        Output filename for the GIF. Default is "prediction_evolution.gif".
    input_length : int, optional
        Length of input sequences. Default is 5.
    target_length : int, optional
        Length of target sequences. Default is 3.
    fps : int, optional
        Frames per second for the GIF. Default is 5.
    """
    # Get the example data
    x_dict, y_dict, file_path = dataset[example_idx]
    
    # Get original data before transforms for plotting
    original_data = np.load(file_path)
    input_1 = original_data['input_1']
    input_2 = original_data['input_2'] 
    input_3 = original_data['input_3']
    target_1 = original_data['target_1']
    target_2 = original_data['target_2']
    
    series_length = len(input_1)
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 4, figsize=(16, 8), constrained_layout=True)
    fig.suptitle(
        f'Time Series Prediction Evolution - Example {example_idx}',
        fontsize=16
    )
    
    # Flatten axes for easier indexing
    ax_input1, ax_input2, ax_input3, ax_pca1 = axes[0]
    ax_pca2, ax_target1, ax_target2, ax_legend = axes[1]
    
    # Set up axes
    time_range = np.arange(series_length)
    
    ax_input1.set_title('Input 1 (Original)')
    ax_input1.set_xlim(0, series_length-1)
    ax_input1.set_ylim(np.min(input_1)*1.1, np.max(input_1)*1.1)
    
    ax_input2.set_title('Input 2 (Original)')
    ax_input2.set_xlim(0, series_length-1)
    ax_input2.set_ylim(np.min(input_2)*1.1, np.max(input_2)*1.1)
    
    ax_input3.set_title('Input 3 (Original)')
    ax_input3.set_xlim(0, series_length-1)
    ax_input3.set_ylim(np.min(input_3)*1.1, np.max(input_3)*1.1)
    
    ax_pca1.set_title('PCA 1 (Transformed)')
    ax_pca1.set_xlim(0, series_length-1)
    ax_pca1.set_ylim(
        np.min(x_dict['pca_1'].numpy())*1.1,
        np.max(x_dict['pca_1'].numpy())*1.1
    )
    
    ax_pca2.set_title('PCA 2 (Transformed)')
    ax_pca2.set_xlim(0, series_length-1)
    ax_pca2.set_ylim(
        np.min(x_dict['pca_2'].numpy())*1.1,
        np.max(x_dict['pca_2'].numpy())*1.1
    )
    
    ax_target1.set_title('Target 1 (True vs Predicted)')
    ax_target1.set_xlim(0, series_length-1)
    ax_target1.set_ylim(np.min(target_1)*1.1, np.max(target_1)*1.1)
    
    ax_target2.set_title('Target 2 (True vs Predicted)')
    ax_target2.set_xlim(0, series_length-1)
    ax_target2.set_ylim(np.min(target_2)*1.1, np.max(target_2)*1.1)
    
    # Create legend
    ax_legend.axis('off')
    ax_legend.text(0.1, 0.8, 'Legend:', fontsize=12, weight='bold')
    ax_legend.text(0.1, 0.64, '● History (α=0.5)', fontsize=10, alpha=0.5)
    ax_legend.text(0.1, 0.48, '● Input Window (α=1.0)', fontsize=10, color='C0')
    ax_legend.text(0.1, 0.32, '● PCA inputs (α=1.0)', fontsize=10, color='C3')
    ax_legend.text(0.1, 0.16, '● True Target (α=1.0)', fontsize=10, color='C2')
    ax_legend.text(0.1, 0.0, '● Predicted (α=1.0)', fontsize=10, color='C1')
    
    model.eval()
    
    def animate(frame):
        # Clear all axes
        for ax in [
            ax_input1,
            ax_input2,
            ax_input3,
            ax_pca1,
            ax_pca2,
            ax_target1,
            ax_target2
        ]:
            ax.clear()
        
        # Current time step
        current_time = frame + input_length
        
        # Plot input signals
        # History with alpha=0.5
        if current_time - input_length > 0:
            ax_input1.plot(time_range[:current_time-input_length], 
                          input_1[:current_time-input_length], 
                          'C0', alpha=0.5, linewidth=1)
            ax_input2.plot(time_range[:current_time-input_length], 
                          input_2[:current_time-input_length], 
                          'C0', alpha=0.5, linewidth=1)
            ax_input3.plot(time_range[:current_time-input_length], 
                          input_3[:current_time-input_length], 
                          'C0', alpha=0.5, linewidth=1)
        
        # Current input window with alpha=1.0
        input_start = current_time - input_length
        input_end = current_time
        ax_input1.plot(time_range[input_start:input_end], 
                      input_1[input_start:input_end], 
                      'C0', alpha=1.0, linewidth=2)
        ax_input2.plot(time_range[input_start:input_end], 
                      input_2[input_start:input_end], 
                      'C0', alpha=1.0, linewidth=2)
        ax_input3.plot(time_range[input_start:input_end], 
                      input_3[input_start:input_end], 
                      'C0', alpha=1.0, linewidth=2)
        
        # Plot PCA signals
        # History with alpha=0.5
        if current_time - input_length > 0:
            ax_pca1.plot(time_range[:current_time-input_length], 
                        x_dict['pca_1'][:current_time-input_length], 
                        'C3', alpha=0.5, linewidth=1)
            ax_pca2.plot(time_range[:current_time-input_length], 
                        x_dict['pca_2'][:current_time-input_length], 
                        'C3', alpha=0.5, linewidth=1)
        
        # Current PCA window with alpha=1.0
        ax_pca1.plot(time_range[input_start:input_end], 
                    x_dict['pca_1'][input_start:input_end], 
                    'C3', alpha=1.0, linewidth=2)
        ax_pca2.plot(time_range[input_start:input_end], 
                    x_dict['pca_2'][input_start:input_end], 
                    'C3', alpha=1.0, linewidth=2)
        
        # Plot target signals
        # History with alpha=0.5
        if current_time > 0:
            ax_target1.plot(time_range[:current_time], 
                           target_1[:current_time], 
                           'C2', alpha=0.5, linewidth=1, label='True')
            ax_target2.plot(time_range[:current_time], 
                           target_2[:current_time], 
                           'C2', alpha=0.5, linewidth=1, label='True')
        
        # Make prediction for current window
        if current_time + target_length <= series_length:
            # Create input tensor for prediction
            input_chunk = TensorDict({
                'pca_1': x_dict['pca_1'][input_start:input_end],
                'pca_2': x_dict['pca_2'][input_start:input_end]
            }, batch_size=[])
            
            # Create batch with single example
            batch_chunk = TensorDict({
                'x': input_chunk,
                'y': TensorDict({}, batch_size=[])  # Empty targets for prediction
            }, batch_size=[])
            
            # Add batch dimension and move to device
            batch_chunk = batch_chunk.unsqueeze(0).to(device)
            
            with torch.no_grad():
                predictions = model(batch_chunk)
            
            # Extract predictions
            pred_target1 = predictions['target_1'].squeeze().cpu().numpy()
            pred_target2 = predictions['target_2'].squeeze().cpu().numpy()
            
            # Plot true targets in prediction window with alpha=1.0
            pred_time_range = time_range[
                current_time:current_time+target_length
            ]
            ax_target1.plot(pred_time_range, 
                           target_1[current_time:current_time+target_length], 
                           'C2', alpha=1.0, linewidth=2)
            ax_target2.plot(pred_time_range, 
                           target_2[current_time:current_time+target_length], 
                           'C2', alpha=1.0, linewidth=2)
        
            # Plot predictions with alpha=1.0
            ax_target1.plot(pred_time_range, pred_target1, 
                           'C1', alpha=1.0, linewidth=2, label='Predicted')
            ax_target2.plot(pred_time_range, pred_target2, 
                           'C1', alpha=1.0, linewidth=2, label='Predicted')
            
        # Set titles and limits
        ax_input1.set_title(f'Input 1 (t={current_time})')
        ax_input1.set_xlim(0, series_length-1)
        ax_input1.set_ylim(np.min(input_1)*1.1, np.max(input_1)*1.1)
        
        ax_input2.set_title(f'Input 2 (t={current_time})')
        ax_input2.set_xlim(0, series_length-1)
        ax_input2.set_ylim(np.min(input_2)*1.1, np.max(input_2)*1.1)
        
        ax_input3.set_title(f'Input 3 (t={current_time})')
        ax_input3.set_xlim(0, series_length-1)
        ax_input3.set_ylim(np.min(input_3)*1.1, np.max(input_3)*1.1)
        
        ax_pca1.set_title(f'PCA 1 (t={current_time})')
        ax_pca1.set_xlim(0, series_length-1)
        ax_pca1.set_ylim(
            np.min(x_dict['pca_1'].numpy())*1.1,
            np.max(x_dict['pca_1'].numpy())*1.1
        )
        
        ax_pca2.set_title(f'PCA 2 (t={current_time})')
        ax_pca2.set_xlim(0, series_length-1)
        ax_pca2.set_ylim(
            np.min(x_dict['pca_2'].numpy())*1.1,
            np.max(x_dict['pca_2'].numpy())*1.1
        )
        
        ax_target1.set_title(f'Target 1 (t={current_time})')
        ax_target1.set_xlim(0, series_length-1)
        ax_target1.set_ylim(np.min(target_1)*1.1, np.max(target_1)*1.1)
        
        ax_target2.set_title(f'Target 2 (t={current_time})')
        ax_target2.set_xlim(0, series_length-1)
        ax_target2.set_ylim(np.min(target_2)*1.1, np.max(target_2)*1.1)
        
        # Add grid to all plots
        for ax in [
            ax_input1,
            ax_input2,
            ax_input3,
            ax_pca1,
            ax_pca2,
            ax_target1,
            ax_target2
        ]:
            ax.grid(True, alpha=0.3)
            ax.set_xlabel('Time')
    
    # Calculate number of frames
    max_frames = series_length - input_length - target_length + 1
    
    # Create animation
    anim = animation.FuncAnimation(
        fig, animate, frames=max_frames, interval=1000//fps, repeat=True
    )
    
    # Save as GIF
    logger.info(f"Creating GIF with {max_frames} frames")
    anim.save(filename, writer='pillow', fps=fps)
    logger.info(f"GIF saved as {filename}")
    
    plt.close(fig)


def demonstrate_prediction_gif():
    """
    Demonstrate the prediction GIF creation using the trained model.
    """
    logger.info("Creating example GIF")

    # Load the trained model
    if not os.path.exists('best_model.pth'):
        logger.error("No trained model found.")
        return
    
    # Determine device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    
    # Load model
    checkpoint = torch.load('best_model.pth', map_location=device)
    hyperparams = checkpoint['hyperparams']
    
    model = TimeSeriesForecastingModel(
        input_dim=2,
        hidden_dim=hyperparams['hidden_dim'],
        num_layers=hyperparams['num_layers']
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Create transforms
    pca_transform = PCATransform("pca_model.joblib")
    scaler_transform = StandardScalerTransform("scaler_model.joblib")
    composed_transform = transforms.Compose([pca_transform, scaler_transform])
    
    # Create test dataset
    test_data_dir = os.path.join("data", "test")
    test_dataset = TimeSeriesDataset(
        test_data_dir,
        transform=composed_transform
    )
    
    # Create GIF for first test example
    create_prediction_gif(
        model=model,
        dataset=test_dataset,
        device=device,
        example_idx=0,
        filename="prediction_evolution_example_0.gif",
        fps=5
    )
    
    logger.info("Prediction GIF demonstration complete!")


# =====================
# Main Execution  
# =====================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="Number of workers for DataLoader (default is 0)"
    )
    parser.add_argument(
        "--num_trials",
        type=int,
        default=20,
        help="Number of trials for Optuna (default is 20)"
    )
    args = parser.parse_args()

    num_workers = args.num_workers
    num_trials = args.num_trials

    # Determine device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    
    logger.info(f"Using device: {device}")
    
    # Generate data
    if not os.path.exists("data"):
        generate_synthetic_timeseries_data()
    
    # Fit preprocessing models
    if not os.path.exists("pca_model.joblib"):
        fit_and_save_pca(os.path.join("data", "train"), n_components=2)
    
    if not os.path.exists("scaler_model.joblib"):
        fit_and_save_scaler(os.path.join("data", "train"))
    
    # Create composed transforms: PCA followed by StandardScaler
    pca_transform = PCATransform("pca_model.joblib")
    scaler_transform = StandardScalerTransform("scaler_model.joblib")
    
    composed_transform = transforms.Compose([
        pca_transform,
        scaler_transform
    ])
    
    # Run hyperparameter optimization
    logger.info("Running hyperparameter optimization")
    best_params = run_hyperparameter_optimization(
        device=device,
        composed_transform=composed_transform,
        n_trials=num_trials,
        n_workers=num_workers
    )
    
    # Train final model with best hyperparameters
    final_model, test_loss = train_final_model(
        device=device,
        best_params=best_params,
        composed_transform=composed_transform,
        num_workers=num_workers
    )

    demonstrate_prediction_gif()
    
    logger.info("Optimization complete")
    logger.info(f"Best hyperparameters: {best_params}")
    logger.info(f"Final test loss: {test_loss:.6f}")
    logger.info("TensorBoard logs saved in 'runs/' directory")
    logger.info("Best model saved as 'best_model.pth'")
    logger.info("Example prediction GIF saved")
