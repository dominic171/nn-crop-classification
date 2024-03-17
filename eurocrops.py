"""
PyTorch-based implementation of EuroCrops dataset.
"""

import os
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset


class EuroCrops(ABC, Dataset):
    """
    Abstract base class for the EuroCrops dataset.

    This class provides the basic structure and methods for creating a custom dataset
    for EuroCrops crop classification.

    Args:
        None

    Attributes:
        None

    Methods:
        __len__: Abstract method to get the length of the dataset.
        __getitem__: Abstract method to get an item from the dataset.

    """

    def __init__(self):
        super().__init__()

    @abstractmethod
    def __len__(self):
        pass

    @abstractmethod
    def __getitem__(self, idx):
        pass


class EuroRegion(EuroCrops):
    """
    EuroRegion class for the EuroCrops dataset.

    This class represents a specific region in the EuroCrops dataset and provides
    methods to load and process the data.

    Args:
        mode (str): The mode of the dataset, either 'train' or 'test'.
        train_region (str): The name of the training region.
        test_region (str): The name of the test region.
        path_train_data_h5 (str): The path to the training data HDF5 file.
        path_test_data_h5 (str): The path to the test data HDF5 file.
        path_train_labels_dir (str): The path to the directory containing the training labels.
        path_test_labels_dir (str): The path to the directory containing the test labels.
        label_encoder (LabelEncoder, optional): The label encoder for encoding class labels.
            ATTENTION: Must be passed in test mode. Use LabelEncoder from training dataset.
            Defaults to LabelEncoder().

    Attributes:
        mode (str): The mode of the dataset, either 'train' or 'test'.
        train_region (str): The name of the training region.
        test_region (str): The name of the test region.
        path_train_data_h5 (str): The path to the training data HDF5 file.
        path_test_data_h5 (str): The path to the test data HDF5 file.
        path_train_labels_dir (str): The path to the directory containing the training labels.
        path_test_labels_dir (str): The path to the directory containing the test labels.
        label_encoder (LabelEncoder): The label encoder for encoding class labels.
        train_df (pd.DataFrame): The training data.
        train_labels_df (pd.DataFrame): The training labels.
        test_df (pd.DataFrame): The test data.
        test_labels_df (pd.DataFrame): The test labels.
        label_map (dict): A mapping of crop group IDs to crop group names.
        nclasses (int): The number of unique crop group IDs.
            Includes all labels from both training and test regions.
        sequence_length (int): The length of the data sequences.
            Here: 52 weeks i.e. the length of the satellite time series.
        input_dim (int): The dimension of the input data.
            Here: 13 i.e. the number of satellite bands.
        length (int): The length of the dataset.
            Specific to the mode of the dataset.

    Methods:
        __len__: Returns the length of the dataset.
        __getitem__: Returns an item from the dataset.
        map_labels: Maps encoded labels to crop group names.

    """

    def __init__(
        self,
        mode: str,
        train_region: str,
        test_region: str,
        path_train_data_h5: str,
        path_test_data_h5: str,
        path_train_labels_dir: str,
        path_test_labels_dir: str,
        label_encoder: LabelEncoder = LabelEncoder(),
    ):
        super().__init__()

        self.mode = mode
        self.train_region = train_region
        self.test_region = test_region
        self.path_train_data_h5 = path_train_data_h5
        self.path_test_data_h5 = path_test_data_h5
        self.path_train_labels_dir = path_train_labels_dir
        self.path_test_labels_dir = path_test_labels_dir
        self.label_encoder = label_encoder

        # Load training data and labels
        hdf = pd.HDFStore(self.path_train_data_h5, mode="r")
        self.train_df = hdf.get(train_region)
        hdf.close()

        train_label_path = os.path.join(
            self.path_train_labels_dir, f"demo_eurocrops_{train_region[1:]}.csv"
        )
        self.train_labels_df = pd.read_csv(
            train_label_path, index_col="recno", usecols=["recno", "crpgrpc", "crpgrpn"]
        )
        print("Finished loading training data and labels")

        # Load test data and labels
        hdf = pd.HDFStore(self.path_test_data_h5, mode="r")
        self.test_df = hdf.get(test_region)
        hdf.close()

        test_label_path = os.path.join(
            self.path_test_labels_dir, f"demo_eurocrops_{test_region[1:]}.csv"
        )
        self.test_labels_df = pd.read_csv(
            test_label_path, index_col="recno", usecols=["recno", "crpgrpc", "crpgrpn"]
        )
        print("Finished loading test data and labels")

        # Resampling to weekly values
        def mean_of_lists(data):
            if len(data) == 0:
                return np.nan
            else:
                return np.nanmean(np.array(data.tolist()), axis=0).tolist()

        all_labels_df = pd.concat([self.train_labels_df, self.test_labels_df], axis=0)
        self.label_map = dict(zip(all_labels_df["crpgrpc"], all_labels_df["crpgrpn"]))

        # Resampling to weekly values
        train_df_transpose = self.train_df.T.copy()
        train_df_transpose.index = pd.to_datetime(train_df_transpose.index)
        train_df_transpose = train_df_transpose.resample("W-SUN").apply(
            mean_of_lists
        )  # resample to weekly values (Sunday)
        train_df_transpose.ffill(inplace=True)
        train_df_transpose.bfill(inplace=True)
        self.train_df = train_df_transpose.T.copy()

        test_df_transpose = self.test_df.T.copy()
        test_df_transpose.index = pd.to_datetime(test_df_transpose.index)
        test_df_transpose = test_df_transpose.resample("W").apply(mean_of_lists)
        test_df_transpose.ffill(inplace=True)
        test_df_transpose.bfill(inplace=True)
        self.test_df = test_df_transpose.T.copy()

        self.train_labels_df.drop("crpgrpn", axis=1, inplace=True)
        self.test_labels_df.drop("crpgrpn", axis=1, inplace=True)

        print("Finished resampling data")

        # Encoding class labels (to integers from 0 to n_classes - 1)
        if mode == "train":
            self.label_encoder.fit(all_labels_df["crpgrpc"].unique())
        # if test mode, pass label encoder from training dataset as argument

        self.train_labels_df["class_label"] = self.label_encoder.transform(
            self.train_labels_df["crpgrpc"]
        )
        self.test_labels_df["class_label"] = self.label_encoder.transform(
            self.test_labels_df["crpgrpc"]
        )

        print("Finished encoding labels")

        # Select data and labels according to mode
        if mode == "train":
            self.data = self.train_df
            self.labels = self.train_labels_df["class_label"]
        elif mode == "test":
            self.data = self.test_df
            self.labels = self.test_labels_df["class_label"]
        else:
            raise ValueError("Mode must be either 'train' or 'test'")

        self.nclasses = len(all_labels_df["crpgrpc"].unique())
        self.sequence_length = self.data.shape[1]
        assert self.sequence_length == 52
        self.input_dim = len(self.data.iloc[0].iloc[0])
        assert self.input_dim == 13

        self.length = len(self.data)

        print("Finished initializing dataset")

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        recno = torch.tensor(self.labels.index[idx], dtype=torch.long)
        X = torch.tensor(
            self.data.iloc[idx].tolist(), dtype=torch.float32
        ).T  # transpose for expected input shape of models
        y = torch.tensor(self.labels.iloc[idx], dtype=torch.long)
        return recno, X, y

    def map_labels(self, encoded_labels):
        """
        Maps encoded labels to crop group names.

        Args:
            input_labels (list): The encoded labels to be mapped.

        Returns:
            list: The mapped crop group names.

        """
        input_labels = self.label_encoder.inverse_transform(encoded_labels)
        output_labels = [
            self.label_map.get(crop_type_id, "UNKNOWN") for crop_type_id in input_labels
        ]
        return output_labels
