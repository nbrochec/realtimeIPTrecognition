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
import os, sys
import numpy as np
import torch, torchaudio, h5py, random, shutil
import torchaudio.transforms as Taudio
import torch.nn.functional as Fnn

from torch.utils.data import TensorDataset, DataLoader
from externals.pytorch_balanced_sampler.sampler import SamplerFactory

import pandas as pd
import csv
from tqdm import tqdm

import datetime

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

                        waveform, original_sr = torchaudio.load(file_path)

                        if original_sr != self.target_sr:
                            waveform = Taudio.Resample(orig_freq=original_sr, new_freq=self.target_sr)(waveform)
                        
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
    def split_train_validation(base_dir, destination='data/dataset/', train_dir='train', test_dir='test', val_ratio=0.2, val_split='train', csv_filename='dataset_split.csv'):
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
        val_split : str
            On which dataset the validation split will be made.
        csv_filename : str
            The name of the output CSV file.
        """

        train_path = os.path.join(base_dir, train_dir)
        test_path = os.path.join(base_dir, test_dir)
        csv_path = os.path.join(destination, csv_filename)

        if val_split != 'train' and val_split != 'test':
            raise ValueError("val_split must be either 'train' or 'test'.")

        with open(csv_path, mode='w', newline='') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(['file_path', 'label', 'set'])

            # Process train directory
            for root, dirs, files in os.walk(train_path):
                label = os.path.basename(root)
                all_files = [os.path.join(root, f) for f in files if f.lower().endswith(('.wav', '.aiff', '.aif', '.mp3'))]

                if val_split == 'train':
                    # Split into train and validation sets
                    num_files = len(all_files)
                    num_val = int(num_files * val_ratio)
                    val_files = random.sample(all_files, num_val)
                    train_files = list(set(all_files) - set(val_files))

                    for file in train_files:
                        writer.writerow([file, label, 'train'])

                    for file in val_files:
                        writer.writerow([file, label, 'val'])
                else:
                    train_files = list(set(all_files))
                    for file in train_files:
                        writer.writerow([file, label, 'train'])

            # Process test directory
            for root, dirs, files in os.walk(test_path):
                label = os.path.basename(root)
                all_files = [os.path.join(root, f) for f in files if f.lower().endswith(('.wav', '.aiff', '.aif', '.mp3'))]

                if val_split == 'test':
                    # Split into test and validation sets
                    num_files = len(all_files)
                    num_val = int(num_files * val_ratio)
                    val_files = random.sample(all_files, num_val)
                    test_files = list(set(all_files) - set(val_files))

                    for file in test_files:
                        writer.writerow([file, label, 'test'])

                    for file in val_files:
                        writer.writerow([file, label, 'val'])
                else:
                    test_files = list(set(all_files))
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

        train_labels = set(data[data['set'] == 'train']['label'].unique())
        test_labels = set(data[data['set'] == 'test']['label'].unique())
        val_labels = set(data[data['set'] == 'val']['label'].unique())

        if not (train_labels == test_labels == val_labels):
            raise ValueError("Mismatch in labels between train, test, and val sets.")
        
        print("Label validation passed: All sets have the same labels.")

    def get_num_labels_from_csv(csv_file):
        data = pd.read_csv(csv_file)
        return len(data['label'].unique())


class ProcessDataset:
    def __init__(self, set_type, csv_path, target_sr, segment_length, silence_threshold=1e-4, min_silence_len=0.1):
        """
        Initialize the ProcessDataset class.

        Parameters
        ----------
        set_type : str
            The type of dataset to process ('train', 'test', or 'val').
        csv_path : str
            Path to the CSV file containing file paths, labels, and set information.
        target_sr : int
            Target sampling rate for audio files.
        segment_length : int
            The length of each audio segment to be extracted.
        silence_threshold : float
            Threshold below which sound is considered silence.
        min_silence_len : float
            Minimum length of silence to remove.
        """
        self.set_type = set_type
        self.csv_path = csv_path
        self.target_sr = target_sr
        self.segment_length = segment_length
        self.silence_threshold = silence_threshold
        self.min_silence_len = min_silence_len

        self.data = pd.read_csv(self.csv_path)
        
        self.data = self.data[self.data['set'] == self.set_type]
        
        self.label_map = {label: idx for idx, label in enumerate(sorted(self.data['label'].unique()))}

        self.X = []
        self.y = []

        self.process_all_files()

    def remove_silence(self, waveform):
        """
        Remove silence from the audio waveform.

        Parameters
        ----------
        waveform : torch.Tensor
            The audio waveform tensor.

        Returns
        -------
        torch.Tensor
            The waveform with silence removed.
        """
        min_silence_samples = int(self.min_silence_len * self.target_sr)
        amplitude = torch.sqrt(torch.mean(waveform**2, dim=0))
        non_silent_indices = torch.where(amplitude > self.silence_threshold)[0]

        if len(non_silent_indices) == 0:
            return waveform

        start = max(0, non_silent_indices[0] - min_silence_samples)
        end = min(waveform.shape[1], non_silent_indices[-1] + min_silence_samples)
        return waveform[:, start:end]
    
    def process_all_files(self):
        """
        Process all audio files and store them in X and y.
        """
        for _, row in self.data.iterrows():
            file_path = row['file_path']
            label = row['label']
            label = self.label_map[label]

            waveform, original_sr = torchaudio.load(file_path)

            if original_sr != self.target_sr:
                waveform = torchaudio.transforms.Resample(orig_freq=original_sr, new_freq=self.target_sr)(waveform)

            waveform = self.remove_silence(waveform)
            num_samples = waveform.size(1)

            for i in range(0, num_samples, self.segment_length):
                if i + self.segment_length <= num_samples:
                    segment = waveform[:, i:i + self.segment_length]
                else:
                    segment = torch.zeros((waveform.size(0), self.segment_length))
                    segment[:, :num_samples - i] = waveform[:, i:]

                self.X.append(segment)
                self.y.append(label)

        self.X = torch.stack(self.X)
        self.y = torch.tensor(self.y)

    def get_data(self):
        """
        Get the processed data as a TensorDataset.

        Returns
        -------
        TensorDataset
            A TensorDataset containing X and y tensors.
        """
        return TensorDataset(self.X, self.y)

class BalancedDataLoader:
    """
    A class for creating a balanced DataLoader from a PyTorch dataset.

    Parameters
    ----------
    dataset : Dataset
        The PyTorch dataset.
    """
    def __init__(self, dataset, device):
        self.dataset = dataset
        self.num_classes = self._get_num_classes()
        self.batch_size = self.num_classes
        self.device = device

        all_targets = [dataset[i][1].unsqueeze(0) if dataset[i][1].dim() == 0 else dataset[i][1] for i in range(len(dataset))]
        all_targets = torch.cat(all_targets)

        class_idxs = [[] for _ in range(self.num_classes)]

        for i in range(self.num_classes):
            indexes = torch.nonzero(all_targets == i).squeeze()
            class_idxs[i] = indexes.tolist()

        total_samples = len(self.dataset)
        n_batches = total_samples // self.batch_size

        self.batch_sampler = SamplerFactory().get(
            class_idxs=class_idxs,
            batch_size=self.batch_size,
            n_batches=n_batches,
            alpha=1,
            kind='fixed'
        )

    def _get_num_classes(self):
        """
        Determines the number of unique classes in the dataset.

        Returns
        -------
        int
            The number of unique classes.
        """
        all_labels = [label.item() for label in self.dataset.tensors[1]]
        unique_classes = set(all_labels)
        num_unique_classes = len(unique_classes)
        return num_unique_classes
    
    def custom_collate_fn(self, batch):
        segments = []
        labels = []
        
        for segs, lbls in batch:
            segments.extend(segs)
            labels.extend(lbls)

        segments_tensor = torch.stack(segments).to(self.device)
        labels_tensor = torch.tensor(labels).to(self.device)

        return segments_tensor, labels_tensor

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
            batch_sampler=self.batch_sampler,
        )
