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

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='This script launches the training process.')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs.')
    parser.add_argument('--config', type=str, default='v2', help='Model version.')
    parser.add_argument('--gpu', type=int, default=0, help='Specify which GPU to use.')
    parser.add_argument('--device', type=str, default='cpu', help='Specify the hardware on which computation should be performed.')
    parser.add_argument('--sr', type=int, default=24000, help='Sampling rate for downsampling the audio files.')
    parser.add_argument('--augment', type=str, default='pitchshift', help='Specify which augmentations to use.')
    parser.add_argument('--early_stopping', type=int, default=None, help='Number of epochs without improvement before early stopping.')
    parser.add_argument('--reduceLR', type=bool, default=False, help='Reduce learning rate if validation plateaus.')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate.')
    parser.add_argument('--fmin', type=int, default=None, help='Minimum frequency for logmelspec analysis.')
    parser.add_argument('--name', type=str, help='Name of the project.', required=True)
    parser.add_argument('--export_ts', type=bool, default=True, help='Export TorchScript file of the model.')
    parser.add_argument('--segment_overlap', type=bool, default=False, help='Overlap the segment when preparing the datasets.')
    parser.add_argument('--save_logs', type=bool, default=True, help='Save logs to disk.')
    return parser.parse_args()
    
def get_run_dir(run_name):
    """Create runs and the current run directories."""
    cwd = os.getcwd()
    runs = os.path.join(cwd, 'runs')
    if not os.path.exists(runs):
        os.makedirs(runs, exist_ok=True)
    os.makedirs(os.path.join(runs, run_name), exist_ok=True)
    current_run = os.path.join(runs, run_name)
    return current_run

def get_csv_file_path(args):
    """Get CSV file path"""
    name = f'{args.name}_dataset_split.csv'
    cwd = os.path.join(os.getcwd(), 'data', 'dataset')
    csv_file_path = os.path.join(cwd, name)
    return csv_file_path

if __name__ == '__main__':
    args = parse_arguments()
    # GetDevice.get_device(args)

    if args.device:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        print('cuda')

    csv_file_path = get_csv_file_path(args)

    dataPreparator = PrepareData(args, csv_file_path, SEGMENT_LENGTH)
    train_loader, test_loader, val_loader, num_classes = dataPreparator.prepare()

    modelPreparator = PrepareModel(args, num_classes, SEGMENT_LENGTH)
    model = modelPreparator.prepare()

    SaveYAML.save_to_disk(args, num_classes)

    current_run = get_run_dir(args.name)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=args.lr)
    if args.reduceLR == True:
        scheduler = ReduceLROnPlateau(optimizer, 'min', patience=20, factor=0.1, verbose=True)

    augmentations = ApplyAugmentations(args.augment.split(), args.sr)
    aug_nbr = augmentations.get_aug_nbr()

    max_val_loss = np.inf
    early_stopping_threshold = args.early_stopping
    counter = 0
    num_epoch = 0

    best_state = None

    trainer = ModelTrainer(model, loss_fn)

    for epoch in range(args.epochs):
        print(f'Epoch {epoch+1}/{args.epochs}')
        train_loss = trainer.train_epoch(train_loader, optimizer, augmentations, aug_nbr)
        print(f'Training Loss: {train_loss:.4f}')
        val_loss = trainer.validate_epoch(val_loader)
        print(f'Validation Loss: {val_loss:.4f}')

        if args.reduceLR:
            scheduler.step()
        
        if args.early_stopping:
            if val_loss < max_val_loss:
                max_val_loss = val_loss
                counter = 0
                num_epoch = epoch
                best_state = model.state_dict()
            else:
                counter += 1
                if counter >= early_stopping_threshold:
                    print('Early stopping triggered.')
                    break

    if args.early_stopping is not None:
        model.load_state_dict(best_state)

    stkd_mtrs, cm = trainer.test_model(test_loader)

    date = datetime.datetime.now().strftime('%Y%m%d')
    time = datetime.datetime.now().strftime('%H%M%S')
    torch.save(model.state_dict(), f'{current_run}/{args.name}_ckpt_{date}_{time}.pth')
    print(f'Checkpoints has been saved in the {os.path.relpath(current_run)} directory.')

    if args.export_ts:
        scripted_model = torch.jit.script(model)
        scripted_model.save(f'{current_run}/{args.name}_ckpt_{date}_{time}.ts')
        print(f'TorchScript file has been exported to the {os.path.relpath(current_run)} directory.')
    
    if args.save_logs:
        SaveResultsToDisk.save_to_disk(args, stkd_mtrs, cm, date, time, csv_file_path)
        print(f'Results have been save to logs directory.')
