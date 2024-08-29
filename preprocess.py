#############################################################################
# preprocess.py 
# Nicolas Brochec
# TOKYO UNIVERSITY OF THE ARTS
# 東京藝術大学音楽音響創造科
# ERC Reach
# GNU General Public License v3.0
#############################################################################
# Code description:
# Script to preprocess the audio data
#############################################################################

from utils import *
import os, h5py, torch, torchaudio, argparse
import numpy as np

# Argument Parse
parser = argparse.ArgumentParser(description='Preprocess audio data and save as HDF5.')
parser.add_argument('--train_dir', type=str, help='Directory of training samples to preprocess', default='train')
parser.add_argument('--test_dir', type=str, help='Directory of test samples to preprocess', default='test')
parser.add_argument('--val_split', type=str, help='Specify on which dataset the validation split would be made', default='train')
parser.add_argument('--val_ratio', type=float, help='Amount of validation samples', default=0.2)

args = parser.parse_args()

csv_dir = 'data/dataset/'
base_dir = 'data/raw_data/'

if __name__ == '__main__':
    print('Preparing the validation set')
    if args.val_ratio == 'train':
        print(f'Validation dataset will be created from the training dataset based on a {args.val_ratio} ratio.')
    elif args.val_ratio == 'test':
        print(f'Validation dataset will be created from the test dataset based on a {args.val_ratio} ratio.')
    DatasetSplitter.split_train_validation(base_dir=base_dir, destination=csv_dir, train_dir=args.train_dir, test_dir=args.test_dir, val_ratio=args.val_ratio, val_split=args.val_split, csv_filename='dataset_split.csv')
    DatasetValidator.validate_labels(os.path.join(csv_dir, 'dataset_split.csv'))