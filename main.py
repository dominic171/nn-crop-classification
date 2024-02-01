"""
Runs the experiment for the thesis.
"""

import random

import numpy as np
import torch

from trainer import train_on_region


def run_experiments():
    # Emperiment 1:
    # Model: TempCNN
    # Training: Region SI031
    # Testing: Region SI036
    tempcnn_args = {
        "model_name": "TempCNN",
        "train_region": "/SI031",
        "test_region": "/SI036",
        "path_train_data_h5": "data/HDF5s/train/SI_T33TWM_train.h5",
        "path_test_data_h5": "data/HDF5s/test/SI_T33TWM_test.h5",
        "path_train_labels_dir": "data/csv_labels/train",
        "path_test_labels_dir": "data/csv_labels/test",
        "num_epochs": 10,  # set according to the early stopping criterion
        "batch_size": 32,
    }

    train_on_region(device, tempcnn_args)

    # Experiment 2:
    # Model: RNN
    # Training: Region SI031
    # Testing: Region SI036
    rnn_args = {
        "model_name": "RNN",
        "train_region": "/SI031",
        "test_region": "/SI036",
        "path_train_data_h5": "data/HDF5s/train/SI_T33TWM_train.h5",
        "path_test_data_h5": "data/HDF5s/test/SI_T33TWM_test.h5",
        "path_train_labels_dir": "data/csv_labels/train",
        "path_test_labels_dir": "data/csv_labels/test",
        "num_epochs": 10,  # set according to the early stopping criterion
        "batch_size": 32,
    }

    train_on_region(device, rnn_args)

    # Experiment 3:
    # Model: Transformer
    # Training: Region SI031
    # Testing: Region SI036
    transformer_args = {
        "model_name": "Transformer",
        "train_region": "/SI031",
        "test_region": "/SI036",
        "path_train_data_h5": "data/HDF5s/train/SI_T33TWM_train.h5",
        "path_test_data_h5": "data/HDF5s/test/SI_T33TWM_test.h5",
        "path_train_labels_dir": "data/csv_labels/train",
        "path_test_labels_dir": "data/csv_labels/test",
        "num_epochs": 10,  # set according to the early stopping criterion
        "batch_size": 32,
    }

    train_on_region(device, transformer_args)


if __name__ == "__main__":
    # Settings seeds and forcing deterministic/benchmark behavior
    random_seed = 2024
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Set device to GPU if available
    device = torch.device(
        "cuda:0"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using device: {device}")

    run_experiments()
