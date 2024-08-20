#############################################################################
# CNN_arch.py 
# Nicolas Brochec
# TOKYO UNIVERSITY OF THE ARTS
# 東京藝術大学音楽音響創造科
# ERC Reach
# MIT license
#############################################################################
# Code description:
# Different architectures for the RT-IPT-R
#############################################################################

import argparse, sys, math
from CNN_arch import v1, v2, heavy, recurrent
import torch

parser = argparse.ArgumentParser(description='train CNN model for RT-IPT-R')
parser.add_argument('--train_dir', type='str', help='train samples directory')
parser.add_argument('--test_dir', type='str', help='test samples directory')
parser.add_argument('--epochs', type=int, help='number of train epochs')
parser.add_argument('--config', type='str', help='version of the CNN')
parser.add_argument('--gpu', type=int, help='gpu')
parser.add_argument('--log_dir', type=int, help='log directory')
parser.add_argument('--sr', type=int, help='sampling rate')

args = parser.parse_args()

train_dir = args.train_dir
test_dir = args.tset_dir
epochs = args.epochs
config = args.config
gpu = args.gpu
log_dir = args.log_dir
sr = args.sr



