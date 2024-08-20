#############################################################################
# CNN_arch.py 
# Nicolas Brochec
# TOKYO UNIVERSITY OF THE ARTS
# 東京藝術大学音楽音響創造科
# ERC Reach
# MIT license 2024
#############################################################################
# Code description:
# Code to launch model's training
#############################################################################

import argparse, sys, math
from architectures import v1, v2, one_residual, two_residual, transformer
import torch

parser = argparse.ArgumentParser(description='train CNN model for RT-IPT-R')
parser.add_argument('--train_dir', type='str', required=True, help='train samples directory')
parser.add_argument('--test_dir', type='str', required=True, help='test samples directory')
parser.add_argument('--epochs', type=int, help='number of train epochs', default=100)
parser.add_argument('--config', type='str', help='version of the CNN', default='v2')
parser.add_argument('--gpu', type=int, help='gpu', default=0)
parser.add_argument('--log_dir', type=int, help='log directory', default='./log_dir')
parser.add_argument('--sr', type=int, help='sampling rate', default=24000)

args = parser.parse_args()

if not args.train_dir:
    print('Error: train samples directory is mandatory.')
    parser.print_help()
    sys.exit(1)

if not args.test_dir:
    print('Error: test samples directory is mandatory.')
    parser.print_help()
    sys.exit(1)

train_dir = args.train_dir
test_dir = args.tset_dir
epochs = args.epochs
config = args.config
gpu = args.gpu
log_dir = args.log_dir
sr = args.sr





