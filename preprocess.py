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

parser = argparse.ArgumentParser(description='preprocess data')
parser.add_argument('--name', type=str, help='name of the dataset (train or test)', required=True)
parser.add_argument('--dir', type=str, help='directory of samples to preprocess', required=True)
parser.add_argument('--save_dir', type=str, help='directory of the h5 file', required=True)
parser.add_argument('--sr', type=int, help='sampling rate', required=True)

args = parser.parse_args()

# Configuration
sample_rate = 24000
segment_length = 7680  # Nombre de samples par segment
hdf5_file = args.name

# Initialisation de la couche customLogMelSpectrogram
mel_transform = customLogMelSpectrogram(sample_rate=sample_rate)

# Création du fichier HDF5
with h5py.File(hdf5_file, 'w') as h5f:
    for root, dirs, files in os.walk(hdf5_file):
        for file in files:
            if file.endswith('.wav') or file.endswith('.aiff') or file.endswith('.mp3'):
                label = os.path.basename(root)
                file_path = os.path.join(root, file)

                # Charger l'audio
                waveform, sr = torchaudio.load(file_path)

                # Downsampling si nécessaire
                if sr != sample_rate:
                    waveform = torchaudio.transforms.Resample(orig_freq=sr, new_freq=sample_rate)(waveform)

                # Découper en segments de 7680 samples et appliquer Log Mel Spectrogram
                mel_spectrograms = mel_transform.process_segment(waveform, segment_length=segment_length)

                # Sauvegarder chaque segment dans le fichier HDF5
                for i, mel_spec in enumerate(mel_spectrograms):
                    grp = h5f.create_group(f"{label}/{file}_{i * segment_length}")
                    grp.create_dataset('mel_spec', data=mel_spec.numpy())
                    grp.attrs['label'] = label