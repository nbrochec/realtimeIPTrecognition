#############################################################################
# train.py 
# Nicolas Brochec
# TOKYO UNIVERSITY OF THE ARTS
# 東京藝術大学音楽音響創造科
# ERC Reach
# GNU General Public License v3.0
#############################################################################
# Code description:
# Script to train the model
#############################################################################

import argparse
import os
import sys
import pandas as pd
import torch
import torchaudio
from torch.utils.data import DataLoader

from models import *
from utils import *
# Constants
SEGMENT_LENGTH = 7680

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='This script launches the training process.')
    parser.add_argument('--csv_file', type=str, default='dataset_split.csv', help='Path to dataset CSV file.')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs.')
    parser.add_argument('--config', type=str, default='v2', help='CNN configuration version.')
    parser.add_argument('--gpu', type=int, default=0, help='GPU device number.')
    parser.add_argument('--sr', type=int, default=24000, help='Sampling rate of audio files.')
    parser.add_argument('--augment', type=str, default='pitchshift', help='Augmentation methods to apply.')
    parser.add_argument('--early_stopping', type=bool, default=False, help='Use early stopping.')
    parser.add_argument('--reduceLR', type=bool, default=False, help='Reduce learning rate on plateau.')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate.')
    parser.add_argument('--fmin', type=int, default=150, help='Minimum frequency for logmelspec analysis.')
    return parser.parse_args()

def get_device():
    """Automatically select the device."""
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'This script use {device} as torch device.')
    return device

def get_num_labels_from_csv(csv_file_path):
    """Extract the number of unique labels from the CSV file."""
    df = pd.read_csv(csv_file_path)
    return len(df['label'].unique())

def collate_fn(batch, device):
    """Load segments and labels on the torch device"""
    segments, labels = zip(*batch)
    segments = [s.to(device) for s in segments]
    labels = [l.to(device) for l in labels]
    return segments, labels

def prepare_data(args, device):
    """Prepare data loaders."""
    # Define file paths and dataset parameters
    cwd = os.path.join(os.getcwd(), 'data', 'dataset')
    csv_file_path = os.path.join(cwd, args.csv_file)

    # Load datasets
    train_dataset = ProcessDataset('train', csv_file_path, args.sr, SEGMENT_LENGTH)
    test_dataset = ProcessDataset('test', csv_file_path, args.sr, SEGMENT_LENGTH)
    val_dataset = ProcessDataset('val', csv_file_path, args.sr, SEGMENT_LENGTH)

    # Create data loaders
    train_loader = BalancedDataLoader(train_dataset, device).get_dataloader()
    test_loader = DataLoader(test_dataset, batch_size=64, collate_fn=lambda x: collate_fn(x, device))
    val_loader = DataLoader(val_dataset, batch_size=64, collate_fn=lambda x: collate_fn(x, device))

    print('Data successfully loaded into DataLoaders.')

    return train_loader, test_loader, val_loader

def prepare_model(args, device):
    """Prepare the model."""
    # Define file paths
    cwd = os.path.join(os.getcwd(), 'data', 'dataset')
    csv_file_path = os.path.join(cwd, args.csv_file)

    # Get number of labels
    num_labels = get_num_labels_from_csv(csv_file_path)

    # Load model
    model = LoadModel().get_model(args.config, num_labels).to(device)
    summary = ModelSummary(model, num_labels, args.config)
    summary.print_summary()

    # Test model
    tester = ModelTester(model, input_shape=(1, 1, SEGMENT_LENGTH), device=device)
    output = tester.test()

    if output.size(1) != num_labels:
        print("Error: Output dimension does not match the number of classes.")
        sys.exit(1)

    return model

if __name__ == '__main__':
    args = parse_arguments()
    device = get_device()
    train_loader, test_loader, val_loader = prepare_data(args, device)
    model = prepare_model(args, device)
    

# for batch_data, batch_labels in balanced_loader:
#     # Apply augmentations
#     augmented_data = []
#     augmented_labels = []

#     for i in range(batch_data.size(0)):  # Use size(0) to get batch size
#         sample_data = batch_data[i]
#         sample_labels = batch_labels[i]
        
#         augmented_samples = augmentation_instance.apply(sample_data.unsqueeze(0))
#         num_samples = augmented_samples.size(0)

#         augmented_data.append(augmented_samples)
#         augmented_labels.extend([sample_labels] * num_samples)

#     # Concatenate augmented data and labels
#     batch_data = torch.cat(augmented_data, dim=0)
#     batch_labels = torch.tensor(augmented_labels, dtype=torch.long)