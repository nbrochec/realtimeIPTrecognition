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
parser.add_argument('--val_dir', type=str, help='Directory of validation samples, optional', default='val')
parser.add_argument('--sr', type=int, help='Sampling rate', default=24000)

args = parser.parse_args()

csv_dir = 'data/dataset/'
base_dir = 'data/raw_data/'

segment_length = 7680

if __name__ == '__main__':
    print('Preparing the validation set')
    DatasetSplitter.split_train_validation(base_dir=base_dir, destination=csv_dir, train_dir='train', test_dir='test', val_ratio=0.2, csv_filename='dataset_split.csv')

    DatasetValidator.validate_labels(os.path.join(csv_dir, 'dataset_split.csv'))