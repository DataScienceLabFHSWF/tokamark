"""
Demonstration of a data loading, preprocessing and PyTorch model fitting
pipeline.

The following steps are performed:

1. Synthetic Data Generation:
   - Generate a synthetic classification dataset with specified features
     and classes.
   - Split the data into training, validation, and test sets.
   - Introduce NaN values randomly in the validation and test sets.
   - Save the datasets as .npz files organized into respective
     directories.

2. Data Preprocessing:
   - Fit a PCA model to the training data to reduce dimensionality and
     save the model.
   - Fit a SimpleImputer to fill in NaN values based on the training
     data and save the model.

3. Dataset and DataLoader Creation:
   - Define a custom PyTorch Dataset class to load data from the .npz
     files.
   - Implement data transformations using PCA and mean imputation.
   - Create DataLoaders for training, validation, and test sets with
     applied transformations.

4. Neural Network Model Definition:
   - Define a simple feedforward neural network using PyTorch.

5. Model Training
"""

import os
import numpy as np
import torch
import joblib
from sklearn.datasets import make_classification
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


# =====================
# Data Generation
# =====================


def generate_synthetic_data(n_samples=1000, nan_ratio=0.05):
    """
    Generate synthetic classification data, introduce NaNs randomly in
    validation and test sets, and store it in .npz files.

    Parameters
    ----------
    n_samples : int, optional
        Number of samples to generate. Defaults to 1000.
    nan_ratio : float, optional
        Proportion of data to replace with NaN in val and test sets.
        Defaults to 0.05.
    """
    X, y = make_classification(
        n_samples=n_samples,
        n_features=20,
        n_informative=10,
        n_classes=2,
        random_state=42,
    )

    # Shuffle indices and split dataset
    indices = np.arange(n_samples)
    np.random.shuffle(indices)
    train_size = int(0.6 * n_samples)
    val_size = int(0.2 * n_samples)
    train_indices = indices[:train_size]
    val_indices = indices[train_size : train_size + val_size]
    test_indices = indices[train_size + val_size :]

    X_train, y_train = X[train_indices], y[train_indices]
    X_val, y_val = X[val_indices], y[val_indices]
    X_test, y_test = X[test_indices], y[test_indices]

    def introduce_nans(X, nan_ratio):
        n_samples, n_features = X.shape
        n_nans = int(n_samples * n_features * nan_ratio)
        flat_indices = np.random.choice(n_samples * n_features, n_nans, replace=False)
        nan_samples, nan_features = np.unravel_index(
            flat_indices, (n_samples, n_features)
        )
        X[nan_samples, nan_features] = np.nan
        return X

    X_val = introduce_nans(X_val, nan_ratio)
    X_test = introduce_nans(X_test, nan_ratio)

    os.makedirs("data", exist_ok=True)
    os.makedirs(os.path.join("data", "train"), exist_ok=True)
    os.makedirs(os.path.join("data", "val"), exist_ok=True)
    os.makedirs(os.path.join("data", "test"), exist_ok=True)

    for i, (xx, yy) in enumerate(zip(X_train, y_train)):
        np.savez(os.path.join("data", "train", f"x_{i:03}"), xx)
        np.savez(os.path.join("data", "train", f"y_{i:03}"), yy)

    for i, (xx, yy) in enumerate(zip(X_val, y_val)):
        np.savez(os.path.join("data", "val", f"x_{i:03}"), xx)
        np.savez(os.path.join("data", "val", f"y_{i:03}"), yy)

    for i, (xx, yy) in enumerate(zip(X_test, y_test)):
        np.savez(os.path.join("data", "test", f"x_{i:03}"), xx)
        np.savez(os.path.join("data", "test", f"y_{i:03}"), yy)

    print("Data saved in data/ subdirectories")


# =====================
# Preprocessing
# =====================


def fit_and_save_pca(train_data_dir, n_components=2, model_path="pca_model.joblib"):
    """
    Fit PCA to the training data and save the model to disk.

    Parameters
    ----------
    train_data_dir : str
        Directory where the train data files are located.
    n_components : int
        Number of components for PCA.
    model_path : str
        File path to save the fitted PCA model.
    """
    # Gather all training data
    x_train = []
    train_files = sorted(
        [
            f
            for f in os.listdir(train_data_dir)
            if f.startswith("x_") and f.endswith(".npz")
        ]
    )

    for x_file in train_files:
        x = np.load(os.path.join(train_data_dir, x_file))["arr_0"]
        x_train.append(x)

    x_train = np.array(x_train)

    pca = PCA(n_components=n_components)
    pca.fit(x_train)

    joblib.dump(pca, model_path)
    print(f"PCA model saved to {model_path}")


def fit_and_save_imputer(train_data_dir, model_path="imputer.joblib"):
    """
    Fit a mean imputer to the training data and save the model to disk.

    Parameters
    ----------
    train_data_dir : str
        Directory where the train data files are located.
    model_path : str
        File path to save the fitted imputer model.
    """
    # Load training data
    X = []
    for f in sorted(os.listdir(train_data_dir)):
        if not f.startswith("x_"):
            continue
        arr = np.load(os.path.join(train_data_dir, f))["arr_0"]
        X.append(arr)
    X = np.vstack(X)

    imputer = SimpleImputer(strategy="mean")
    imputer.fit(X)

    joblib.dump(imputer, model_path)
    print(f"Imputer model saved to {model_path}")


# =====================
# Dataset and Transforms
# =====================


class SyntheticDataset(Dataset):
    """
    Custom Dataset for loading synthetic data stored in .npz files.

    Parameters
    ----------
    data_dir : str
        Directory where the data files are located.
    transform : callable, optional
        Transform to be applied on a sample. Defaults to None.
    """

    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform

        self.data_files = sorted(
            [
                f
                for f in os.listdir(self.data_dir)
                if f.startswith("x_") and f.endswith(".npz")
            ]
        )

    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, idx):
        x_file = self.data_files[idx]
        x = np.load(os.path.join(self.data_dir, x_file))["arr_0"]

        y_file = x_file.replace("x_", "y_")
        y = np.load(os.path.join(self.data_dir, y_file))["arr_0"]

        x = torch.from_numpy(x).float()
        y = torch.tensor(y, dtype=torch.long)

        if self.transform:
            x, y = self.transform((x, y))

        return x, y


class PCATransform(object):
    """Use a pre-fitted PCA function to transform input data.

    Parameters
    ----------
    pca_model_path : str
        Path to fitted pca model joblib file.
    """

    def __init__(self, pca_model_path):
        self.pca_model = joblib.load(pca_model_path)

    def __call__(self, sample):
        x, y = sample
        x_np = self.pca_model.transform(x.numpy().reshape(1, -1))
        x_np = x_np.squeeze(0)
        x = torch.tensor(x_np, dtype=x.dtype)
        return x, y


class MeanImputerTransform(object):
    """Use a pre-fitted mean imputer to transform input data.

    Parameters
    ----------
    model_path : str
        Path to fitted imputer model joblib file.
    """

    def __init__(self, model_path):
        self.imputer = joblib.load(model_path)

    def __call__(self, sample):
        x, y = sample
        x_np = x.numpy().reshape(1, -1)
        x_imp = self.imputer.transform(x_np).squeeze(0)
        return torch.from_numpy(x_imp).float(), y


def create_dataloaders(batch_size=32, num_workers=6, transform=None):
    """
    Create data loaders for train, validation, and test datasets.

    Parameters
    ----------
    batch_size : int
        Number of samples per batch.
    num_workers : int
        Number of subprocesses to use for data loading.
    transform : callable, optional
        Transform to be applied to a sample. Defaults to None.

    Returns
    -------
    tuple
        Data loaders for train, validation, and test datasets.
    """
    train_data_dir = os.path.join("data", "train")
    val_data_dir = os.path.join("data", "val")
    test_data_dir = os.path.join("data", "test")

    train_dataset = SyntheticDataset(train_data_dir, transform=transform)
    val_dataset = SyntheticDataset(val_data_dir, transform=transform)
    test_dataset = SyntheticDataset(test_data_dir, transform=transform)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
    )
    test_dataloader = DataLoader(
        test_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False
    )

    return train_dataloader, val_dataloader, test_dataloader


# =====================
# Model Definition and Training
# =====================


class NeuralNetwork(nn.Module):
    """
    Simple feedforward neural network.

    Architecture
        - Fully connected layer with 64 units
        - ReLU activation
        - Fully connected output layer with 2 units
    """

    def __init__(self, input_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 2)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x


def train_model(
    device, train_dataloader, val_dataloader, input_size, num_epochs=10, lr=1e-4
):
    """
    Train the NeuralNetwork model.

    Parameters
    ----------
    device : torch.device
        The device to train the model on.
    train_dataloader : DataLoader
        DataLoader for the training data.
    val_dataloader : DataLoader
        DataLoader for the validation data.
    input_size : int
        Size of the input passed to the first layer of the model.
    num_epochs : int, optional
        Number of epochs to train the model.
    lr : float, optional
        Learning rate.
    """
    model = NeuralNetwork(input_size=input_size).to(device)
    criterion = nn.CrossEntropyLoss()
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(num_epochs):
        print(f"Epoch: {epoch}")
        model.train()
        running_loss = 0.0
        for batch_idx, (data, target) in enumerate(train_dataloader):
            data, target = data.to(device), target.to(device)

            outputs = model(data)
            loss = criterion(outputs, target)

            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

            running_loss += loss.item()
        print(f"Train loss: {running_loss/len(train_dataloader):.2e}")

        model.eval()
        running_loss = 0.0
        for batch_idx, (data, target) in enumerate(val_dataloader):
            data, target = data.to(device), target.to(device)

            outputs = model(data)
            loss = criterion(outputs, target)

            running_loss += loss.item()
        print(f"Val loss: {running_loss/len(val_dataloader):.2e}")


# =====================
# Main Execution
# =====================

if __name__ == "__main__":
    # Determine device to train on
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    generate_synthetic_data()

    n_pca_components = 4
    fit_and_save_pca(os.path.join("data", "train"), n_components=n_pca_components)
    fit_and_save_imputer(os.path.join("data", "train"))

    composed_transforms = transforms.Compose(
        [MeanImputerTransform("imputer.joblib"), PCATransform("pca_model.joblib")]
    )

    train_dataloader, val_dataloader, test_dataloader = create_dataloaders(
        num_workers=2, transform=composed_transforms, batch_size=64
    )

    train_model(
        device,
        train_dataloader,
        val_dataloader,
        input_size=n_pca_components,
        num_epochs=100,
        lr=1e-4,
    )
