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
from torch.optim import Adam, AdamW
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter

from models import *
from utils import *

def parse_arguments():

    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='This script launches the training process.')
    parser.add_argument('--name', type=str, help='Name of the project.', required=True)
    parser.add_argument('--device', type=str, default='cpu', help='Specify the hardware on which computation should be performed.')
    parser.add_argument('--gpu', type=int, default=0, help='Specify which GPU to use.')
    parser.add_argument('--config', type=str, default='v2', help='Model version.')
    parser.add_argument('--sr', type=int, default=24000, help='Sampling rate for downsampling the audio files.')
    parser.add_argument('--segment_overlap', type=str, default='False', help='Overlap between audio segments. Increase the data samples by a factor 2.')
    parser.add_argument('--fmin', type=int, default=0, help='Minimum frequency for logmelspec analysis.')
    parser.add_argument('--online_augment', type=str, default='', help='Specify which online augmentations to use.')
    parser.add_argument('--offline_augment', type=str, default='True', help='Use offline augmentations. True or False.')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate.')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs.')
    parser.add_argument('--early_stopping', type=int, default=None, help='Number of epochs without improvement before early stopping.')
    parser.add_argument('--reduce_lr', type=str, default='False', help='Reduce learning rate if validation plateaus. True or False.')
    parser.add_argument('--export_ts', type=str, default='True', help='Export TorchScript file of the model. True or False.')
    parser.add_argument('--padding', type=str, default='minimal', help='Pad the arrays with zeros.')
    parser.add_argument('--save_logs', type=str, default='True', help='Save results and confusion matrix to disk. True or False.')
    parser.add_argument('--batch_size', type=int, default=128, help='Specify batch size.')
    
    args = parser.parse_args()
    
    # Convert string to boolean
    args.segment_overlap = args.segment_overlap.lower() in ['true', '1']
    args.offline_augment = args.offline_augment.lower() in ['true', '1']
    args.reduce_lr = args.reduce_lr.lower() in ['true', '1']
    args.export_ts = args.export_ts.lower() in ['true', '1']
    args.save_logs = args.save_logs.lower() in ['true', '1']

    return args
    # """Parse command-line arguments."""
    # parser = argparse.ArgumentParser(description='This script launches the training process.')
    # parser.add_argument('--name', type=str, help='Name of the project.', required=True)
    # parser.add_argument('--device', type=str, default='cpu', help='Specify the hardware on which computation should be performed.')
    # parser.add_argument('--gpu', type=int, default=0, help='Specify which GPU to use.')
    # parser.add_argument('--config', type=str, default='v2', help='Model version.')
    # parser.add_argument('--sr', type=int, default=24000, help='Sampling rate for downsampling the audio files.')
    # parser.add_argument('--segment_overlap', type=bool, default=False, help='Overlap between audio segments. Increase the data samples by a factor 2.')
    # parser.add_argument('--fmin', type=int, default=0, help='Minimum frequency for logmelspec analysis.')
    # parser.add_argument('--online_augment', type=str, default='', help='Specify which online augmentations to use.')
    # parser.add_argument('--offline_augment', type=bool, default=True, help='Use offline augmentations.')
    # parser.add_argument('--lr', type=float, default=0.001, help='Learning rate.')
    # parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs.')
    # parser.add_argument('--early_stopping', type=int, default=None, help='Number of epochs without improvement before early stopping.')
    # parser.add_argument('--reduce_lr', type=bool, default=False, help='Reduce learning rate if validation plateaus.')
    # parser.add_argument('--export_ts', type=bool, default=True, help='Export TorchScript file of the model.')
    # parser.add_argument('--padding', type=str, default='minimal', help='Pad the arrays with zeros.')
    # parser.add_argument('--save_logs', type=bool, default=True, help='Save results and confusion matrix to disk.')
    # parser.add_argument('--batch_size', type=int, default=128, help='Specify batch size.')
    # return parser.parse_args()
    
def get_run_dir(run_name):
    """Create runs and the current run directories."""
    cwd = os.getcwd()
    runs = os.path.join(cwd, 'runs')
    if not os.path.exists(runs):
        os.makedirs(runs, exist_ok=True)
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

    device = GetDevice.get_device(args)

    csv_file_path = get_csv_file_path(args)

    dataPreparator = PrepareData(args, csv_file_path, SEGMENT_LENGTH)
    train_loader, test_loader, val_loader, num_classes = dataPreparator.prepare()

    modelPreparator = PrepareModel(args, num_classes, SEGMENT_LENGTH)
    model = modelPreparator.prepare()

    loss_fn = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    if args.reduce_lr == True:
        scheduler = ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.1)

    augmenter = AudioOnlineTransforms(args)

    max_val_loss = np.inf
    early_stopping_threshold = args.early_stopping
    counter = 0
    num_epoch = 0

    trainer = ModelTrainer(model, loss_fn, args.device)
    date = datetime.datetime.now().strftime('%Y%m%d')
    time = datetime.datetime.now().strftime('%H%M%S')
    current_run = f'{get_run_dir(args.name)}_{date}_{time}'
    SaveYAML.save_to_disk(args, num_classes, current_run)
    print(f'Run {args.name}_{date}_{time}.')
    
    writer = SummaryWriter(log_dir= f'runs/{args.name}_{date}_{time}')
    args.num_classes = num_classes
    args_dict = vars(args)
    writer.add_text('Hyperparameters', Dict2MDTable.apply(args_dict), 1)

    for epoch in range(args.epochs):
        train_loss = trainer.train_epoch(train_loader, optimizer, augmenter)
        writer.add_scalar('epoch/epoch', epoch, epoch)
        writer.add_scalar('Loss/train', train_loss, epoch)

        val_loss, val_acc, val_pre, val_rec, val_f1 = trainer.validate_epoch(val_loader)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Accuracy/val', val_acc, epoch)
        writer.add_scalar('Precision/val', val_pre, epoch)
        writer.add_scalar('Recall/val', val_rec, epoch)
        writer.add_scalar('F1/val', val_f1, epoch)

        if args.reduce_lr:
            scheduler.step(val_loss)
        
        if val_loss < max_val_loss:
            max_val_loss = val_loss
            counter = 0
            num_epoch = epoch
            best_state = model.state_dict()
            # Save the checkpoints each time a model version is better than the former one
            torch.save(model.state_dict(), f'{current_run}/{args.name}_{date}_{time}.pth')
        else:
            if args.early_stopping:
                counter += 1
                if counter >= early_stopping_threshold:
                    print('Early stopping triggered.')
                    break

    # if args.early_stopping:
    model.load_state_dict(best_state)

    stkd_mtrs, cm = trainer.test_model(test_loader)

    SaveResultsToTensorboard.upload(stkd_mtrs, cm, csv_file_path, writer)
    print(f'Results have been uploaded to tensorboard.')

    SaveResultsToDisk.save_to_disk(args, stkd_mtrs, cm, csv_file_path, current_run)
    print(f'Results and Confusion Matrix have been saved in the logs/{os.path.basename(current_run)} directory.')

    torch.save(model.state_dict(), f'{current_run}/{args.name}_{date}_{time}.pth')
    print(f'Checkpoints has been saved in the {os.path.relpath(current_run)} directory.')

    if args.export_ts:
        model = model.to('cpu')
        scripted_model = torch.jit.script(model)
        scripted_model.save(f'{current_run}/{args.name}_{date}_{time}.ts')
        print(f'TorchScript file has been exported to the {os.path.relpath(current_run)} directory.')


