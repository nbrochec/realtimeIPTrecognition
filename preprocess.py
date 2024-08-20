#############################################################################
# preprocess.py 
# Nicolas Brochec
# TOKYO UNIVERSITY OF THE ARTS
# 東京藝術大学音楽音響創造科
# ERC Reach
# MIT license 2024
#############################################################################
# Code description:
# Preprocess the data
#############################################################################

from utils import AudioDatasetManager, customLogMelSpectrogram
import os, h5py, torch, torchaudio, argparse
import numpy as np
from torch.utils.data import Dataset, DataLoader

torchaudio.set_audio_backend("sox_io")

# Argument Parser
parser = argparse.ArgumentParser(description='Preprocess audio data and save as HDF5.')
parser.add_argument('--train_dir', type=str, help='Directory of training samples to preprocess', required=True)
parser.add_argument('--test_dir', type=str, help='Directory of test samples to preprocess', required=True)
parser.add_argument('--save_dir', type=str, help='Directory to save the HDF5 files', required=True)
parser.add_argument('--sr', type=int, help='Sampling rate', required=True)

args = parser.parse_args()

# Configuration
train_hdf5_file = os.path.join(args.save_dir, 'train_data.h5')
test_hdf5_file = os.path.join(args.save_dir, 'test_data.h5')
segment_length = 7680
mel_transform = customLogMelSpectrogram(sample_rate=args.sr)

def preprocess_and_save(data_dir, hdf5_file):
    with h5py.File(hdf5_file, 'w') as h5f:
        for root, dirs, files in os.walk(data_dir):
            for file in files:
                if file.lower().endswith(('.wav', '.aiff', '.mp3')):
                    label = os.path.basename(root)
                    file_path = os.path.join(root, file)

                    waveform, original_sr = torchaudio.load(file_path)

                    if original_sr != args.sr:
                        waveform = torchaudio.transforms.Resample(orig_freq=original_sr, new_freq=args.sr)(waveform)

                    mel_spectrograms = mel_transform.process_segment(waveform, segment_length=segment_length)

                    for i, mel_spec in enumerate(mel_spectrograms):
                        grp = h5f.create_group(f"{label}/{file}_{i * segment_length}")
                        grp.create_dataset('mel_spec', data=mel_spec.numpy())
                        grp.attrs['label'] = label

preprocess_and_save(args.train_dir, train_hdf5_file)
preprocess_and_save(args.test_dir, test_hdf5_file)