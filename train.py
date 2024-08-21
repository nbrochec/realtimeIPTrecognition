#############################################################################
# train.py 
# Nicolas Brochec
# TOKYO UNIVERSITY OF THE ARTS
# 東京藝術大学音楽音響創造科
# ERC Reach
# GNU General Public License v3.0
#############################################################################
# Code description:
# Train a model
#############################################################################

import argparse, sys, math, os
from architectures import v1, v2, one_residual, two_residual, transformer
import torch, torchaudio
from glob import glob
from os.path import join

from utils import BalancedDataLoader, HDF5Dataset, DirectoryManager

parser = argparse.ArgumentParser(description='train CNN model for RT-IPT-R')
parser.add_argument('--dataset_dir', type=str, help='dataset directory where h5 files are saved', required=True)
parser.add_argument('--epochs', type=int, help='number of train epochs', default=100)
parser.add_argument('--config', type=str, help='version of the CNN', default='v2')
parser.add_argument('--device', type=str, help='gpu device', default='cpu')
parser.add_argument('--gpu', type=int, help='gpu device number', default=0)
parser.add_argument('--log_dir', type=str, help='log directory', default='log_dir')
parser.add_argument('--augment', type=str, help='augmentations', default='pitchshift')
parser.add_argument('--early_stopping', type=bool, help='early stopping', default=False)
parser.add_argument('--reduceLR', type=bool, help='reduce LR on plateau', default=False)
parser.add_argument('--lr', type=float, help='learning rate', default=0.001)

args = parser.parse_args()

if not args.dataset_dir:
    print('Error: dataset directory is required to launch training.')
    print('Please indicate the path of the dataset directory.')
    parser.print_help()
    sys.exit(1)

dataset_dir = args.dataset_dir
epochs = args.epochs
config = args.config
device = args.device
gpu = args.gpu
log_dir = args.log_dir
augment = args.augment
early_stopping = args.early_stopping
reduceLR = args.reduceLR
lr = args.lr

dir_manager = DirectoryManager()
DirectoryManager.ensure_dir_exists(args.log_dir)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Current device: {device}')

train_hdf5_file_path = join(dataset_dir, "train_data.h5")
data_loader_factory = BalancedDataLoader(train_hdf5_file_path)

balanced_loader = data_loader_factory.get_dataloader()
for batch_data, batch_labels in balanced_loader:
    print(f"Batch data shape: {batch_data.shape}")
    print(f"Batch labels shape: {batch_labels.shape}")
    print("Train data samples have been loaded.")
    break

test_hdf5_file_path = join(dataset_dir, "test_data.h5")
test_loader = HDF5Dataset(test_hdf5_file_path)
print("Test data samples have been loaded.")

val_hdf5_file_path = join(dataset_dir, "val_data.h5")
val_loader = HDF5Dataset(val_hdf5_file_path)
print("validation data samples have been loaded.")

