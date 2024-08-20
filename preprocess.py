#############################################################################
# preprocess.py 
# Nicolas Brochec
# TOKYO UNIVERSITY OF THE ARTS
# 東京藝術大学音楽音響創造科
# ERC Reach
# GNU General Public License v3.0
#############################################################################
# Code description:
# Preprocess the data
#############################################################################

from utils import customLogMelSpectrogram, ensure_dir_exists, check_hdf5_sanity, remove_silence
import os, h5py, torch, torchaudio, argparse
import numpy as np
from torch.utils.data import Dataset, DataLoader

# torchaudio.set_audio_backend("sox_io")

# Argument Parse
parser = argparse.ArgumentParser(description='Preprocess audio data and save as HDF5.')
parser.add_argument('--train_dir', type=str, help='Directory of training samples to preprocess', required=True)
parser.add_argument('--test_dir', type=str, help='Directory of test samples to preprocess', required=True)
parser.add_argument('--save_dir', type=str, help='Directory to save the HDF5 files')
parser.add_argument('--sr', type=int, help='Sampling rate', default=24000)

args = parser.parse_args()
ensure_dir_exists(args.save_dir)

# Configuration
train_hdf5_file = os.path.join(args.save_dir, 'train_data.h5')
test_hdf5_file = os.path.join(args.save_dir, 'test_data.h5')
segment_length = 7680
mel_transform = customLogMelSpectrogram(sample_rate=args.sr)

# Check if train and test datasets have the same class names

def preprocess_and_save(data_dir, hdf5_file):
    with h5py.File(hdf5_file, 'w') as h5f:
        for root, dirs, files in os.walk(data_dir):
            for file in files:
                if file.lower().endswith(('.wav', '.aiff', '.aif', '.mp3')):
                    label = os.path.basename(root)
                    file_path = os.path.join(root, file)

                    waveform, original_sr = torchaudio.load(file_path)

                    if original_sr != args.sr:
                        waveform = torchaudio.transforms.Resample(orig_freq=original_sr, new_freq=args.sr)(waveform)
                    
                    waveform = remove_silence(waveform, sample_rate=args.sr)

                    mel_spectrograms = mel_transform.process_segment(waveform, segment_length=segment_length)

                    for i, mel_spec in enumerate(mel_spectrograms):
                        grp = h5f.create_group(f"{label}/{file}_{i * segment_length}")
                        grp.create_dataset('mel_spec', data=mel_spec.numpy())
                        grp.attrs['label'] = label

if __name__ == '__main__':
    print('Preprocessing training data samples...')
    preprocess_and_save(args.train_dir, train_hdf5_file)
    print('Preprocessing test data samples...')
    preprocess_and_save(args.test_dir, test_hdf5_file)

    print('Sanity check of h5 files')
    check_hdf5_sanity(train_hdf5_file)
    check_hdf5_sanity(test_hdf5_file)