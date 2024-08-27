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
from tqdm import tqdm
import torch
import torchaudio
import numpy as np

from torch.utils.data import DataLoader
from torch.optim import Adam
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau

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
    parser.add_argument('--device', type=str, default='cpu', help='Device.')
    parser.add_argument('--sr', type=int, default=24000, help='Sampling rate of audio files.')
    parser.add_argument('--augment', type=str, default='pitchshift', help='Augmentation methods to apply.')
    parser.add_argument('--early_stopping', type=bool, default=False, help='Use early stopping.')
    parser.add_argument('--reduceLR', type=bool, default=False, help='Reduce learning rate on plateau.')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate.')
    parser.add_argument('--fmin', type=int, default=150, help='Minimum frequency for logmelspec analysis.')
    parser.add_argument('--save_dir', type=str, default='checkpoints', help='Directory in which checkpoints are saved.')
    return parser.parse_args()

def get_device(device_name, gpu):
    """Automatically select the device."""
    if device_name != 'cpu' and gpu == 0:
        device = torch.device('cuda:0')
        print(f'This script uses {device} as the torch device.')
    elif device_name != 'cpu' and gpu != 0:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
        device = torch.device(f'{device_name}:{gpu}')
        print(f'This script uses {device} as the torch device.')
    else:
        device = torch.device('cpu')
        print(f'This script uses {device} as the torch device.')

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

def get_save_dir(dir):
    cwd = os.getcwd()
    save_dir = os.path.join(cwd, dir)
    return save_dir

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
    train_loader = BalancedDataLoader(train_dataset.get_data(), device).get_dataloader()
    test_loader = DataLoader(test_dataset.get_data(), batch_size=64, collate_fn=lambda x: collate_fn(x, device))
    val_loader = DataLoader(val_dataset.get_data(), batch_size=64, collate_fn=lambda x: collate_fn(x, device))

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

    model = ModelInit(model).initialize()
    
    return model

def train_epoch(model, loader, optimizer, loss_fn, augmentations, aug_number, device):
    model.train()
    running_loss = 0.0

    for data, targets in tqdm(loader, desc="Training", leave=False):
        optimizer.zero_grad()

        # Apply augmentations
        augmented_data = augmentations.apply(data)
        new_data = torch.cat((data, augmented_data), dim=0)
        all_targets = torch.flatten(targets.repeat(aug_nbr+1, 1))

        outputs = model(new_data)
        loss = loss_fn(outputs, all_targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * data.size(0)
    
    return running_loss / len(loader.dataset)

def validate_epoch(model, loader, loss_fn, device):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for data, targets in tqdm(loader, desc="Validation", leave=False):
            data, targets = data.to(device), targets.to(device)
            outputs = model(data)
            loss = loss_fn(outputs, targets)
            running_loss += loss.item() * data.size(0)
    
    return running_loss / len(loader.dataset)

if __name__ == '__main__':
    args = parse_arguments()
    device = get_device(args.device, args.gpu)
    train_loader, test_loader, val_loader = prepare_data(args, device)
    model = prepare_model(args, device)
    save_dir = get_save_dir(args.save_dir)

    cross_entropy = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=args.lr)
    if args.reduceLR == True:
        scheduler = ReduceLROnPlateau(optimizer, 'min', patience=20, factor=0.1, verbose=True)

    augmentations = ApplyAugmentations(args.augment.split(), args.sr, device)
    aug_nbr = len(args.augment.split())

    max_val_loss = np.inf
    early_stopping_threshold = 10
    counter = 0
    num_epoch = 0
    save_dir = get_save_dir(args.save_dir)
    for epoch in range(args.epochs):
        print(f'Epoch {epoch+1}/{args.epochs}')
        train_loss = train_epoch(model, train_loader, optimizer, cross_entropy, augmentations, aug_nbr, device)
        print(f'Training Loss: {train_loss:.4f}')
        val_loss = validate_epoch(model, val_loader, cross_entropy, device)
        print(f'Validation Loss: {val_loss:.4f}')

        if args.reduceLR == True:
            scheduler.step()
        
        if args.early_stopping == True:
            if val_loss < max_val_loss:
                max_val_loss = val_loss
                counter = 0
                torch.save(model.state_dict(), f'{save_dir}/best_model_{epoch}.pth')
                num_epoch = epoch
            else:
                counter += 1
                if counter >= early_stopping_threshold:
                    print('Early stopping triggered.')
                    break

    if args.early_stopping:
        model.load_state_dict(torch.load(f'{save_dir}/best_model_{num_epoch}.pth'))

    test_loss = validate_epoch(model, test_loader, cross_entropy, device)