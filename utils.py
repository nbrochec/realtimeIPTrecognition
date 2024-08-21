#############################################################################
# utils.py 
# Nicolas Brochec
# TOKYO UNIVERSITY OF THE ARTS
# 東京藝術大学音楽音響創造科
# ERC Reach
# GNU General Public License v3.0
#############################################################################
# Code description:
# Utilities defition
#############################################################################

from os.path import join, dirname, basename, abspath, normpath, isdir, exists, relpath
from os import listdir, makedirs, walk
from glob import glob
import os
import torch, torchaudio, h5py, random, shutil
import torchaudio.transforms as Taudio
import torch.nn.functional as Fnn

class customLogMelSpectrogram():
    def __init__(self, sample_rate=24000, n_fft=2048, win_length=None, hop_length=512, n_mels=128, f_min=150, f_max=None):
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.win_length = win_length if win_length is not None else n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.f_min = f_min
        self.f_max = f_max if f_max is not None else sample_rate / 2
        
        self.mel_scale = Taudio.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=self.n_fft,
            win_length=self.win_length,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
            f_min=self.f_min,
            f_max=self.f_max,
            power=2.0
        )

        self.amplitude_to_db = Taudio.AmplitudeToDB(stype='power', top_db=80.0)
    
    def __call__(self, waveform):
        mel_spec = self.mel_scale(waveform)
        mel_spec_db = self.amplitude_to_db(mel_spec)
        mel_spec_db = Fnn.normalize(mel_spec_db, dim=1)
        mel_spec_db = Fnn.normalize(mel_spec_db, dim=2)
        return mel_spec_db
    
    def pad_segment(self, segment, current_segment_length, segment_length):
        pad_length = segment_length - current_segment_length
        segment = Fnn.pad(segment, (0, pad_length), mode='constant', value=0)
        return segment
        
    def process_segment(self, waveform, segment_length=7680):
        """Divides the waveform into segments of a fixed length and applies the Mel transformation."""
        segments = []
        total_length = waveform.size(1)
        # print(total_length)

        if total_length < segment_length:
            pad_length = segment_length - total_length
            waveform = Fnn.pad(waveform, (0, pad_length), mode='constant', value=0)
            total_length = waveform.size(1)
        
        for start in range(0, total_length - segment_length + 1, segment_length):
            segment = waveform[:, start:start + segment_length]
            mel_spec_db = self.__call__(segment)[..., :-1] # manually remove the last frame to keep 15 frames only
            segments.append(mel_spec_db)

        stacked_segments = torch.stack(segments, dim=0)
        return stacked_segments
    
def remove_silence(waveform, sample_rate, silence_threshold=1e-4, min_silence_len=0.1):
    min_silence_samples = int(min_silence_len * sample_rate)
    amplitude = torch.sqrt(torch.mean(waveform**2, dim=0))
    non_silent_indices = torch.where(amplitude > silence_threshold)[0]

    if len(non_silent_indices) == 0:
        return waveform

    start = max(0, non_silent_indices[0] - min_silence_samples)
    end = min(waveform.shape[1], non_silent_indices[-1] + min_silence_samples)
    return waveform[:, start:end]

def ensure_dir_exists(directory):
    if not exists(directory):
        makedirs(directory)
        print(f'{directory} have been created.')

def check_hdf5_sanity(hdf5_file, h5_file_name):
    try:
        with h5py.File(hdf5_file, 'r') as h5f:
            # print("HDF5 file opened successfully.")

            labels = list(h5f.keys())
            num_classes = len(labels)
            if num_classes == 0:
                print("Error: No classes found in the HDF5 file.")
                return False
            elif num_classes == 1:
                print("Error: Only one class found in the HDF5 file. Should be more than one.")
                return False
            # else:
            #     print(f"Number of classes: {num_classes}")

            for label in labels:
                class_group = h5f[label]
                for file_key in class_group.keys():
                    dataset = class_group[file_key]['samples']
                    for i in range(dataset.shape[0]):
                        if dataset[i].shape != (7680,):
                            print(f"Error: Segment {i} in dataset '{file_key}' in class '{label}' does not have the expected shape (7680,).")
                            print(f"Actual shape: {dataset[i].shape}")
                            return False

            print(f"{h5_file_name} file passed all checks.")
            return True

    except Exception as e:
        print(f"Error opening or processing HDF5 file: {e}")
        return False

def check_matching_labels(train_hdf5_file, val_hdf5_file, test_hdf5_file):
    try:
        with h5py.File(train_hdf5_file, 'r') as train_h5f, h5py.File(test_hdf5_file, 'r') as test_h5f, h5py.File(val_hdf5_file, 'r') as val_h5f:
            # Get the unique labels from both HDF5 files
            train_labels = set(train_h5f.keys())
            test_labels = set(test_h5f.keys())
            val_labels = set(val_h5f.keys())

            # Compare the labels
            if train_labels == test_labels == val_labels:
                print("Labels are consistent between the train, val and test HDF5 files.")
                return True
            else:
                print("Error: Labels do not match.")
                print(f"Labels in train file but not in validation and test files: {train_labels - val_labels - test_labels}")
                print(f"Labels in validation file but not in train and test files: {val_labels - train_labels - test_labels}")
                print(f"Labels in test file but not in train and validation files: {test_labels - train_labels - val_labels}")
                return False

    except Exception as e:
        print(f"Error opening or processing HDF5 files: {e}")
        return False

def split_train_validation(train_dir, val_ratio=0.2, val_dir_name='val_dir'):
    parent_dir = dirname(train_dir)
    val_dir = join(parent_dir, val_dir_name)

    all_files = []
    for root, dirs, files in walk(train_dir):
        for file in files:
            if file.endswith('.wav') or file.endswith('.aiff') or file.endswith('.mp3'):
                all_files.append(join(root, file))

    num_files = len(all_files)
    num_val_files = int(num_files * val_ratio)

    val_files = random.sample(all_files, num_val_files)
    
    if not exists(val_dir):
        makedirs(val_dir)

    for file_path in val_files:
        rel_path = relpath(file_path, start=train_dir)
        val_file_path = join(val_dir, rel_path)
        val_file_dir = dirname(val_file_path)

        if not exists(val_file_dir):
            makedirs(val_file_dir)

        shutil.move(file_path, val_file_path)

    print(f"Moved {num_val_files} files to validation set.")