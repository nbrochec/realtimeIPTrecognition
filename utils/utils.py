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

import pandas as pd
import csv

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
        # else:
        #     print(f'{directory} already exists.')

class PreprocessAndSave:
    def __init__(self, base_dir, data_dir, destination, target_sr, segment_length, silence_threshold=1e-4, min_silence_len=0.1):
        """
        Initializes the PreprocessAndSave class.

        Parameters
        ----------
        base_dir : str
            Base directory containing the data.
        data_dir : str
            Directory containing audio files relative to base_dir.
        destination : str
            Directory where processed files will be saved.
        target_sr : int
            Target sampling rate for resampling audio files.
        segment_length : int
            Length of each audio segment to be saved.
        silence_threshold : float
            Threshold below which audio is considered silent.
        min_silence_len : float
            Minimum length of silence to be removed.
        """
        self.base_dir = base_dir
        self.data_dir = data_dir
        self.destination = destination
        self.target_sr = target_sr
        self.segment_length = segment_length
        self.silence_threshold = silence_threshold
        self.min_silence_len = min_silence_len

        self.data_dir_path = join(self.base_dir, self.data_dir)
        self.processed_data_dir = self.destination
        self.hdf5_file = os.path.join(self.destination, f'{self.data_dir}_data.h5')

    def check_for_audio_files(self):
        """
        Checks if there are audio files in the specified directories.

        Raises
        ------
        RuntimeError
            If no audio files are found in the specified directories.
        """
        audio_files_found = False
        for root, dirs, files in os.walk(self.data_dir_path):
            for file in files:
                if file.lower().endswith(('.wav', '.aiff', '.aif', '.mp3')):
                    audio_files_found = True
                    break
            if audio_files_found:
                break
        
        if not audio_files_found:
            raise RuntimeError(f"No audio files found in {self.data_dir_path}. Please check the directory and try again.")

    def silence_remover(self, waveform, sample_rate):
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

    def preprocess_and_save(self):
        """
        Processes audio files from the directory and saves them to an HDF5 file.
        """

        self.check_for_audio_files()  # Check if there are audio files before processing
        DirectoryManager.ensure_dir_exists(self.processed_data_dir)

        with h5py.File(self.hdf5_file, 'w') as h5f:
            for root, dirs, files in os.walk(self.data_dir_path):
                for file in files:
                    if file.lower().endswith(('.wav', '.aiff', '.aif', '.mp3')):
                        label = os.path.basename(root)
                        file_path = os.path.join(root, file)

                        # Load audio file
                        waveform, original_sr = torchaudio.load(file_path)

                        # Resample if necessary
                        if original_sr != self.target_sr:
                            waveform = Taudio.Resample(orig_freq=original_sr, new_freq=self.target_sr)(waveform)
                        
                        # Remove silence
                        waveform = self.silence_remover(waveform, sample_rate=self.target_sr)

                        num_samples = waveform.size(1)
                        
                        for i, start in enumerate(range(0, num_samples - self.segment_length + 1, self.segment_length)):
                            segment = waveform[:, start:start + self.segment_length]
                            grp = h5f.create_group(f"{label}/{file}_{i * self.segment_length}")
                            grp.create_dataset('samples', data=segment.numpy())
                            grp.attrs['label'] = label

        print(f"Processed data saved to {self.hdf5_file}.")

class DatasetSplitter:
    @staticmethod
    def split_train_validation(base_dir, destination='data/dataset/', train_dir='train', test_dir='test', val_ratio=0.2, csv_filename='dataset_split.csv'):
        """
        Splits the dataset into training, validation, and test sets, and saves the information in a CSV file.

        Parameters
        ----------
        base_dir : str
            The base directory containing the data.
        train_dir : str
            The name of the training directory.
        test_dir : str
            The name of the test directory.
        val_ratio : float
            The ratio of the validation set to the total training dataset.
        csv_filename : str
            The name of the output CSV file.
        """

        train_path = os.path.join(base_dir, train_dir)
        test_path = os.path.join(base_dir, test_dir)
        csv_path = os.path.join(destination, csv_filename)

        with open(csv_path, mode='w', newline='') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(['file_path', 'label', 'set'])

            # Process train directory
            for root, dirs, files in os.walk(train_path):
                label = os.path.basename(root)
                all_files = [os.path.join(root, f) for f in files if f.lower().endswith(('.wav', '.aiff', '.aif', '.mp3'))]

                # Split into train and validation sets
                num_files = len(all_files)
                num_val = int(num_files * val_ratio)
                val_files = random.sample(all_files, num_val)
                train_files = list(set(all_files) - set(val_files))

                for file in train_files:
                    writer.writerow([file, label, 'train'])

                for file in val_files:
                    writer.writerow([file, label, 'val'])

            # Process test directory
            for root, dirs, files in os.walk(test_path):
                label = os.path.basename(root)
                test_files = [os.path.join(root, f) for f in files if f.lower().endswith(('.wav', '.aiff', '.aif', '.mp3'))]

                for file in test_files:
                    writer.writerow([file, label, 'test'])

        print(f"CSV file '{csv_filename}' created successfully in {base_dir}.")

class DatasetValidator:
    @staticmethod
    def validate_labels(csv_file):
        """
        Validates that the train, test, and val sets have the same unique labels.

        Parameters
        ----------
        csv_file : str
            Path to the CSV file containing dataset information.

        Raises
        ------
        ValueError
            If the labels in the train, test, and val sets do not match in number or name.
        """
        data = pd.read_csv(csv_file)

        # Get unique labels for each set
        train_labels = set(data[data['set'] == 'train']['label'].unique())
        test_labels = set(data[data['set'] == 'test']['label'].unique())
        val_labels = set(data[data['set'] == 'val']['label'].unique())

        # Check if all sets have the same labels
        if not (train_labels == test_labels == val_labels):
            raise ValueError("Mismatch in labels between train, test, and val sets.")
        
        print("Label validation passed: All sets have the same labels.")

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
   