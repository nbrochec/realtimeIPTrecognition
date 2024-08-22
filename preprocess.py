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
parser.add_argument('--val_dir', type=str, help='Directory of validation samples, automatically created, optional', default='val')
parser.add_argument('--sr', type=int, help='Sampling rate', default=24000)

args = parser.parse_args()

h5_dir = 'data/processed_data/'
base_dir = 'data/raw_data/'

train_hdf5_file_path = os.path.join(h5_dir, 'train_data.h5')
test_hdf5_file_path = os.path.join(h5_dir, 'test_data.h5')
val_hdf5_file_path = os.path.join(h5_dir, 'val_data.h5')
segment_length = 7680

if __name__ == '__main__':
    print('Preparing the validation set')
    DatasetSplitter.split_train_validation(base_dir=base_dir, train_name=args.train_dir, val_name=args.val_dir, val_ratio=0.2)

    train_preprocessor = PreprocessAndSave(base_dir=base_dir,
                                           data_dir=args.train_dir, 
                                           destination=h5_dir, 
                                           target_sr=args.sr, 
                                           segment_length=segment_length,
                                           silence_threshold=1e-4,
                                           min_silence_len=0.1)

    val_preprocessor = PreprocessAndSave(base_dir=base_dir,
                                         data_dir=args.val_dir, 
                                         destination=h5_dir,  
                                         target_sr=args.sr, 
                                         segment_length=segment_length,
                                         silence_threshold=1e-4,
                                         min_silence_len=0.1)

    test_preprocessor = PreprocessAndSave(base_dir=base_dir,
                                          data_dir=args.test_dir, 
                                          destination=h5_dir,  
                                          target_sr=args.sr, 
                                          segment_length=segment_length,
                                          silence_threshold=1e-4,
                                          min_silence_len=0.1)

    print('Preprocessing training data samples...')
    train_preprocessor.preprocess_and_save()

    print('Preprocessing validation data samples...')
    val_preprocessor.preprocess_and_save()

    print('Preprocessing test data samples...')
    test_preprocessor.preprocess_and_save()

    print('Sanity check of h5 files')
    HDF5Checker.check_sanity(train_hdf5_file_path)
    HDF5Checker.check_sanity(val_hdf5_file_path)
    HDF5Checker.check_sanity(test_hdf5_file_path)
    HDF5LabelChecker.check_matching_labels(train_hdf5_file_path, test_hdf5_file_path, val_hdf5_file_path)