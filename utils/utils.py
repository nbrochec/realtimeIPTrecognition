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
        """
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f'{directory} has been created.')

class DatasetSplitter:
    @staticmethod
    def split_train_validation(base_dir, destination='data/dataset/', train_dir='train', test_dir='test', val_ratio=0.2, val_split='train', name='title'):
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
        csv_filename = f'{name}_dataset_split.csv'

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

    def get_num_classes_from_csv(csv_file):
        data = pd.read_csv(csv_file)
        return len(data['label'].unique())


class ProcessDataset:
    def __init__(self, set_type, csv_path, target_sr, segment_overlap, segment_length, silence_threshold=1e-4, min_silence_len=0.1):
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
        self.segment_overlap = segment_overlap # implement rollof 

        self.data = pd.read_csv(self.csv_path)
        
        self.data = self.data[self.data['set'] == self.set_type]
        
        self.label_map = {label: idx for idx, label in enumerate(sorted(self.data['label'].unique()))}

        self.X = []
        self.y = []

        self.process_all_files()

    def remove_silence(self, waveform):
        """
        Remove silence from the audio waveform.
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

            if self.segment_overlap == True and self.set_type == 'train':
                for i in range(0, num_samples, self.segment_length//2):
                    if i + self.segment_length <= num_samples:
                        segment = waveform[:, i:i + self.segment_length]
                    else:
                        segment = torch.zeros((waveform.size(0), self.segment_length))
                        segment[:, :num_samples - i] = waveform[:, i:]

                    self.X.append(segment)
                    self.y.append(label)
            else:
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
        self.num_classes = self.get_num_classes()
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

    def get_num_classes(self):
        """
        Determines the number of unique classes in the dataset.
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
        """
        return DataLoader(
            self.dataset,
            batch_sampler=self.batch_sampler,
        )

class PrepareData:
    """
    Prepare datasets.
    """
    def __init__(self, args, csv_file_path, seg_len, device):
        self.args = args
        self.csv = csv_file_path
        self.seg_len = seg_len
        self.device = device

    def prepare(self):
        num_classes = DatasetValidator.get_num_classes_from_csv(self.csv)
        train_dataset = ProcessDataset('train', self.csv, self.args.sr, self.args.segment_overlap, self.seg_len)
        test_dataset = ProcessDataset('test', self.csv, self.args.sr, self.args.segment_overlap, self.seg_len)
        val_dataset = ProcessDataset('val', self.csv, self.args.sr, self.args.segment_overlap, self.seg_len)

        train_loader = BalancedDataLoader(train_dataset.get_data(), self.device).get_dataloader()
        test_loader = DataLoader(test_dataset.get_data(), batch_size=64)
        val_loader = DataLoader(val_dataset.get_data(), batch_size=64)

        print('Data successfully loaded into DataLoaders.')

        return train_loader, test_loader, val_loader, num_classes
