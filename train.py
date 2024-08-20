#############################################################################
# train.py 
# Nicolas Brochec
# TOKYO UNIVERSITY OF THE ARTS
# 東京藝術大学音楽音響創造科
# ERC Reach
# MIT license 2024
#############################################################################
# Code description:
# Train a model
#############################################################################

import argparse, sys, math, os
from architectures import v1, v2, one_residual, two_residual, transformer
import torch, torchaudio
from glob import glob
from os.path import join

parser = argparse.ArgumentParser(description='train CNN model for RT-IPT-R')
parser.add_argument('--datasets_dir', type=str, required=True, help='dataset directory where h5 files are saved')
parser.add_argument('--epochs', type=int, help='number of train epochs', default=100)
parser.add_argument('--config', type=str, help='version of the CNN', default='v2')
parser.add_argument('--device', type=int, help='gpu device', default='cuda')
parser.add_argument('--gpu', type=int, help='gpu device number', default=0)
parser.add_argument('--log_dir', type=int, help='log directory', default='./log_dir')
parser.add_argument('--augment', type=str, help='augmentations', default='pitchshift')

args = parser.parse_args()

if not args.datasets_dir:
    print('Error: datasets directory is required to launch training.')
    print('Please indicate the path of the datasets directory.')
    parser.print_help()
    sys.exit(1)

datasets_dir = args.train_dir
epochs = args.epochs
config = args.config
device = args.device
gpu = args.gpu
log_dir = args.log_dir
augment = args.augment

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Current device: {device}')