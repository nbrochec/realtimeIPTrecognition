#############################################################################
# utils.py 
# Nicolas Brochec
# TOKYO UNIVERSITY OF THE ARTS
# 東京藝術大学音楽音響創造科
# ERC Reach
# GNU General Public License v3.0
#############################################################################
# Code description:
# Implement global utility functions
#############################################################################

from os.path import join, dirname, basename, abspath, normpath, isdir, exists, relpath
from os import listdir, makedirs, walk
from glob import glob
from pathlib import Path
import os
import numpy as np
import torch, torchaudio, h5py, random, shutil
import torchaudio.transforms as Taudio
import torch.nn.functional as Fnn

from torch.utils.data import Dataset, DataLoader
from externals.pytorch_balanced_sampler.sampler import SamplerFactory

class SilenceRemover:
    def __init__(self, silence_threshold=1e-4, min_silence_len=0.1):
        """
        Initializes the SilenceRemover class with specified parameters.
        
        Parameters
        ----------
        silence_threshold : float
            Threshold of the silence in RMS amplitude.
        min_silence_len : float
            Minimum length of the silence to be removed (percentage).
        """
        self.silence_threshold = silence_threshold
        self.min_silence_len = min_silence_len
    
    def remove_silence(self, waveform, sample_rate):
        """
        Removes silence from an audio segment.

        Parameters
        ----------
        waveform : torch.Tensor
            Audio waveform tensor.
        sample_rate : int
            Audio sampling rate.
        
        Returns
        -------
        torch.Tensor
            Waveform with silence removed.
        """
        min_silence_samples = int(self.min_silence_len * sample_rate)
        amplitude = torch.sqrt(torch.mean(waveform**2, dim=0))
        non_silent_indices = torch.where(amplitude > self.silence_threshold)[0]

        if len(non_silent_indices) == 0:
            return waveform

        start = max(0, non_silent_indices[0] - min_silence_samples)
        end = min(waveform.shape[1], non_silent_indices[-1] + min_silence_samples)
        return waveform[:, start:end]

class DirectoryManager:
    @staticmethod
    def ensure_dir_exists(directory):
        """
        Ensures that the directory exists. If not, creates it.
        
        Parameters
        ----------
        directory : str
            Directory path.
        """
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f'{directory} has been created.')
        else:
            print(f'{directory} already exists.')


class HDF5Checker:
    @staticmethod
    def check_sanity(hdf5_file):
        """
        Checks the sanity of the HDF5 file.
        
        Parameters
        ----------
        hdf5_file : str
            Path to the HDF5 file.
        
        Returns
        -------
        bool
            True if the file passed all checks, False otherwise.
        """
        h5_file_name = os.path.basename(hdf5_file)
        
        try:
            with h5py.File(hdf5_file, 'r') as h5f:
                labels = list(h5f.keys())
                num_classes = len(labels)

                if num_classes == 0:
                    print("Error: No classes found in the HDF5 file.")
                    return False
                elif num_classes == 1:
                    print("Error: Only one class found in the HDF5 file. Should be more than one.")
                    return False

                for label in labels:
                    class_group = h5f[label]
                    for file_key in class_group.keys():
                        dataset = class_group[file_key]['samples']
                        for i in range(dataset.shape[0]):
                            if dataset[i].shape != (7680,):
                                print(f"Error: Segment {i} in dataset '{file_key}' in class '{label}' does not have the expected shape (7680,).")
                                print(f"Actual shape: {dataset[i].shape}")
                                return False

                print(f"{h5_file_name} file passed all checks.")
                return True

        except Exception as e:
            print(f"Error opening or processing HDF5 file: {e}")
            return False

class HDF5LabelChecker:
    @staticmethod
    def check_matching_labels(train_hdf5_file, val_hdf5_file, test_hdf5_file):
        """
        Checks if the labels are consistent across train, validation, and test HDF5 files.

        Parameters
        ----------
        train_hdf5_file : str
            Path to the training HDF5 file.
        val_hdf5_file : str
            Path to the validation HDF5 file.
        test_hdf5_file : str
            Path to the test HDF5 file.

        Returns
        -------
        bool
            True if labels are consistent across all files, False otherwise.
        """
        try:
            with h5py.File(train_hdf5_file, 'r') as train_h5f, \
                 h5py.File(val_hdf5_file, 'r') as val_h5f, \
                 h5py.File(test_hdf5_file, 'r') as test_h5f:
                
                # Get the unique labels from all HDF5 files
                train_labels = set(train_h5f.keys())
                val_labels = set(val_h5f.keys())
                test_labels = set(test_h5f.keys())

                # Compare the labels
                if train_labels == val_labels == test_labels:
                    print("Labels are consistent between the train, val, and test HDF5 files.")
                    return True
                else:
                    print("Error: Labels do not match.")
                    print(f"Labels in train file but not in validation and test files: {train_labels - val_labels - test_labels}")
                    print(f"Labels in validation file but not in train and test files: {val_labels - train_labels - test_labels}")
                    print(f"Labels in test file but not in train and validation files: {test_labels - train_labels - val_labels}")
                    return False

        except Exception as e:
            print(f"Error opening or processing HDF5 files: {e}")
            return False

class DatasetSplitter:
    @staticmethod
    def split_train_validation(train_dir, val_ratio=0.2, val_dir_name='val_dir'):
        """
        Splits the training dataset into training and validation sets.

        Parameters
        ----------
        train_dir : str
            Path to the training directory.
        val_ratio : float
            Ratio of the validation set to the total dataset.
        val_dir_name : str
            Name of the validation directory to be created.

        Returns
        -------
        None
        """
        parent_dir = dirname(train_dir)
        val_dir = join(parent_dir, val_dir_name)

        # Check if the validation directory already exists and is not empty
        if os.path.exists(val_dir) and any(Path(val_dir).rglob("*.wav")):
            print(f"Validation directory '{val_dir}' already exists and is not empty.")
            print("Skipping sample separation.")
            return

        # Collect all audio files from the training directory
        all_files = []
        for root, dirs, files in os.walk(train_dir):
            for file in files:
                if file.endswith(('.wav', '.aiff', '.mp3')):
                    all_files.append(join(root, file))

        # Calculate the number of validation files
        num_files = len(all_files)
        num_val_files = int(num_files * val_ratio)

        # Select a random sample of files for validation
        val_files = random.sample(all_files, num_val_files)
        
        # Create the validation directory if it does not exist
        if not exists(val_dir):
            os.makedirs(val_dir)

        # Move selected files to the validation directory
        for file_path in val_files:
            rel_path = relpath(file_path, start=train_dir)
            val_file_path = join(val_dir, rel_path)
            val_file_dir = dirname(val_file_path)

            if not exists(val_file_dir):
                os.makedirs(val_file_dir)

            shutil.move(file_path, val_file_path)

        print(f"Moved {num_val_files} files to validation set.")

class HDF5Dataset(Dataset):
    def __init__(self, hdf5_file, device=None):
        """
        Initializes the HDF5Dataset.

        Parameters
        ----------
        hdf5_file : str
            Path to the HDF5 file.
        device : torch.device, optional
            Device on which to load the data (e.g., 'cuda' or 'cpu').
        """
        self.device = device
        self.data = []
        self.labels = []

        with h5py.File(hdf5_file, 'r') as h5f:
            for label in h5f.keys():
                group = h5f[label]
                for key in group.keys():
                    sample = group[key]['samples'][:]
                    self.data.append(sample)
                    self.labels.append(label)

        self.data = np.array(self.data)
        self.data = torch.tensor(self.data, dtype=torch.float32)

        # Convert labels to integers
        self.labels = np.array(self.labels)
        _, label_to_int = np.unique(self.labels, return_inverse=True)
        self.labels = torch.tensor(label_to_int, dtype=torch.long)

        if self.device:
            self.data = self.data.to(self.device)
            self.labels = self.labels.to(self.device)

    def __len__(self):
        return self.data.size(0)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]
    
    def get_num_classes(self):
        return len(torch.unique(self.labels))

class BalancedDataLoader:
    """
    A class for creating a balanced DataLoader from an HDF5 dataset.

    This class loads a dataset from an HDF5 file and sets up a DataLoader that ensures
    balanced batches for training. The balanced batches are created using a fixed sampling strategy
    to include samples from each class in every batch.

    pytorch_balanced_sampler have been implemented by Karl Hornlund
    https://github.com/khornlund/pytorch-balanced-sampler

    Parameters
    ----------
    hdf5_file : str
        Path to the HDF5 file containing the dataset.
    device : torch.device, optional
        Device on which to load the data (e.g., 'cuda' or 'cpu').

    Attributes
    ----------
    dataset : HDF5Dataset
        The dataset loaded from the HDF5 file.
    num_classes : int
        The number of unique classes in the dataset.
    batch_sampler : Sampler
        The sampler used to generate balanced batches.

    Methods
    -------
    __init__(hdf5_file)
        Initializes the BalancedDataLoader with the specified HDF5 file.
    get_dataloader()
        Returns a DataLoader instance configured with the balanced batch sampler.
    """
    
    def __init__(self, hdf5_file, device=None):
        """
        Initializes the BalancedDataLoader.

        Parameters
        ----------
        hdf5_file : str
            Path to the HDF5 file.
        device : torch.device, optional
            Device on which to load the data (e.g., 'cuda' or 'cpu').
        """
        self.dataset = HDF5Dataset(hdf5_file, device=device)
        self.device = device
        self.num_classes = len(set(self.dataset.labels.tolist()))  # Number of classes from labels
        batch_size = self.num_classes * 2 

        # Initialize lists to store indexes for each class
        class_idxs = [[] for _ in range(self.num_classes)]
        
        # Populate the class index lists
        for i in range(self.num_classes):
            indexes = torch.nonzero(self.dataset.labels == i).squeeze()
            class_idxs[i] = indexes.tolist()

        # Calculate the number of batches
        total_samples = self.dataset.data.size(0)
        n_batches = total_samples // batch_size

        # Create a balanced batch sampler using the SamplerFactory
        self.batch_sampler = SamplerFactory().get(
            class_idxs=class_idxs,
            batch_size=batch_size,
            n_batches=n_batches,
            alpha=1,
            kind='fixed'
        )

    def get_dataloader(self):
        """
        Returns a DataLoader with the balanced batch sampler.

        Returns
        -------
        DataLoader
            A DataLoader instance configured with the balanced batch sampler.
        """
        return DataLoader(
            self.dataset,
            batch_sampler=self.batch_sampler
        )
   