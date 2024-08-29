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
import random
import string
import datetime

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
    parser.add_argument('--early_stopping', type=bool, default=True, help='Use early stopping.')
    parser.add_argument('--reduceLR', type=bool, default=False, help='Reduce learning rate on plateau.')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate.')
    parser.add_argument('--fmin', type=int, default=150, help='Minimum frequency for logmelspec analysis.')
    parser.add_argument('--name', type=str, default='untitled', help='Name of the run.', required=True)
    parser.add_argument('--export_ts', type=bool, default=True, help='Export TorchScript file of the model.')
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
    
def get_run_dir(run_name):
    """Create runs directory where checkpoints will be saved"""
    cwd = os.getcwd()
    runs = os.path.join(cwd, 'runs')

    if not os.path.exists(runs):
        os.makedirs(runs, exist_ok=True)

    os.makedirs(os.path.join(runs, run_name), exist_ok=True)

    current_run = os.path.join(runs, run_name)
    checkpoints = os.path.join(current_run, 'checkpoints')

    if not os.path.exists(checkpoints):
        os.makedirs(checkpoints, exist_ok=True)

    return current_run, checkpoints

def get_csv_file_path(args):
    """Get CSV file path"""
    cwd = os.path.join(os.getcwd(), 'data', 'dataset')
    csv_file_path = os.path.join(cwd, args.csv_file)

    return csv_file_path

if __name__ == '__main__':
    args = parse_arguments()
    device = get_device(args.device, args.gpu)

    csv_file_path = get_csv_file_path(args)

    dataPreparator = PrepareData(args, csv_file_path, SEGMENT_LENGTH, device)
    train_loader, test_loader, val_loader, num_classes = dataPreparator.prepare()

    modelPreparator = PrepareModel(args, num_classes, SEGMENT_LENGTH, device)
    model = modelPreparator.prepare()

    current_run, ckpt = get_run_dir(args.name)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=args.lr)
    if args.reduceLR == True:
        scheduler = ReduceLROnPlateau(optimizer, 'min', patience=20, factor=0.1, verbose=True)

    augmentations = ApplyAugmentations(args.augment.split(), args.sr, device)
    aug_nbr = augmentations.get_aug_nbr()

    max_val_loss = np.inf
    early_stopping_threshold = 10
    counter = 0
    num_epoch = 0

    trainer = ModelTrainer(model, loss_fn, device)

    for epoch in range(args.epochs):
        print(f'Epoch {epoch+1}/{args.epochs}')
        train_loss = trainer.train_epoch(train_loader, optimizer, augmentations, aug_nbr)
        print(f'Training Loss: {train_loss:.4f}')
        val_loss = trainer.validate_epoch(val_loader)
        print(f'Validation Loss: {val_loss:.4f}')

        if args.reduceLR == True:
            scheduler.step()
        
        if args.early_stopping == True:
            if val_loss < max_val_loss:
                max_val_loss = val_loss
                counter = 0
                num_epoch = epoch
                timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
                torch.save(model.state_dict(), f'{ckpt}/{args.name}_ckpt_{epoch}_{timestamp}.pth')
            else:
                counter += 1
                if counter >= early_stopping_threshold:
                    print('Early stopping triggered.')
                    break

    if args.early_stopping:
        model.load_state_dict(torch.load(f'{ckpt}/{args.name}_ckpt_{num_epoch}_{timestamp}.pth'))

    acc, pre, rec, f1, test_loss = trainer.test_model(test_loader)

    if args.early_stopping == False:
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        torch.save(model.state_dict(), f'{current_run}/{args.name}_ckpt_{timestamp}.pth')
        print(f'Checkpoints has been saved in the {current_run} directory.')
    else:
        torch.save(model.state_dict(), f'{current_run}/{args.name}_ckpt_{timestamp}.pth')
        print(f'Checkpoints has been saved in the {current_run} directory.')

    if args.export_ts:
        scripted_model = torch.jit.script(model)
        scripted_model.save(os.path.join(current_run, f'{args.name}_{timestamp}.ts'))
        print(f'TorchScript file has been exported to the {current_run} directory.')