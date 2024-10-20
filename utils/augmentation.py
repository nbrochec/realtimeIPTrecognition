#############################################################################
# augmentations.py 
# Nicolas Brochec
# TOKYO UNIVERSITY OF THE ARTS
# 東京藝術大学音楽音響創造科
# ERC Reach
# GNU General Public License v3.0
#############################################################################
# Code description:
# Implement data augmentation methods
#############################################################################

import torch, librosa
from torch_audiomentations import Compose, PitchShift, AddColoredNoise, Shift, PolarityInversion, HighPassFilter, LowPassFilter
import numpy as np

class AudioOnlineTransforms:
    def __init__(self, args):
        self.sr = args.sr
        self.device = args.device
        self.online_augment = args.online_augment.split()

    def pitch_shift(self):
        return PitchShift(min_transpose_semitones=-12.0, max_transpose_semitones=12.0, sample_rate=self.sr, p=1, output_type='tensor')
    
    def shift(self):
        return Shift(rollover=False, p=1, output_type='tensor')
    
    def add_noise(self):
        return AddColoredNoise(p=1, output_type='tensor')
    
    def polarity_inversion(self):
        return PolarityInversion(p=1, output_type='tensor')
    
    def highpassfilter(self):
        return HighPassFilter(p=1, output_type='tensor')
    
    def lowpassfilter(self):
        return LowPassFilter(p=1, output_type='tensor')
    
    def none(self):
        return lambda x: x  

    def build_transform_pipeline(self):
        aug_dict = {
            'None': self.none(),
            'pitchshift': self.pitch_shift(),
            'timeshift': self.shift(),
            'polarityinversion': self.polarity_inversion(),
            'hpf': self.highpassfilter(),
            'lpf': self.lowpassfilter(),
            'addnoise': self.add_noise()
        }

        transforms = []
        for augmentation in self.online_augment:
            if augmentation in aug_dict:
                if np.random.rand() < 0.5:
                    transforms.append(aug_dict[augmentation])

        return Compose(transforms)

    def __call__(self, data):
        transform_pipeline = self.build_transform_pipeline()
        data = transform_pipeline(data, sample_rate=self.sr)
        return data
    
class AudioOfflineTransforms:
    def __init__(self, args):
        self.sr = args.sr
        self.device = args.device

    def tensor_to_array(self, data):
        data_numpy = data.cpu().squeeze(1).detach().numpy()
        return data_numpy

    def custom_detune(self, data):
        random_tuning = np.random.uniform(-100, 100) + 440
        change_tuning = librosa.A4_to_tuning(random_tuning)
        data = librosa.effects.pitch_shift(data, sr=self.sr, n_steps=change_tuning, bins_per_octave=12)
        return data

    def custom_gaussnoise(self, data):
        n_channels, length = data.shape
        data = data + np.random.normal(0, 0.01, size=(n_channels, length)) * np.random.randn(n_channels, length)
        return data
    
    def lb_timestretch(self, data):
        length = data.shape[1]
        rate = np.random.uniform(0.9, 1.1)
        data = librosa.effects.time_stretch(data, rate=rate)
        return data[:, :length]
    
    def shift(self, data):
        transform = Shift(rollover=False, p=1)
        return transform(data, sample_rate=self.sr)
    
    def pad_or_trim(self, data, original_size):
        current_size = data.shape[1]
        if current_size > original_size:
            data = data[..., :original_size]
        elif current_size < original_size:
            padding = (0, original_size - current_size)
            data = np.pad(data, pad_width=((0, 0), (0, original_size - current_size)), mode='constant', constant_values=0)
        return data
    
    def __call__(self, data):
        data_numpy = self.tensor_to_array(data)
        original_size = data_numpy.shape[1]

        aug_dict = {
            'detune': self.custom_detune,
            'gaussnoise': self.custom_gaussnoise,
            'timestretch': self.lb_timestretch,
            # 'shift': self.shift,
        }

        detuned = aug_dict['detune'](data_numpy)
        noised = aug_dict['gaussnoise'](data_numpy)
        stretched = aug_dict['timestretch'](data_numpy)
        # shifted = aug_dict['shift'](data_numpy)

        # Ensure all outputs have the same size
        aug_detuned = self.pad_or_trim(detuned, original_size)
        aug_noised = self.pad_or_trim(noised, original_size)
        aug_stretched = self.pad_or_trim(stretched, original_size)
        # aug_shifted = self.pad_or_trim(shifted, original_size)

        aug1 = torch.tensor(aug_detuned).to(torch.float32)
        aug2 = torch.tensor(aug_noised).to(torch.float32)
        aug3 = torch.tensor(aug_stretched).to(torch.float32)
        # aug4 = torch.tensor(aug_shifted).to(torch.float32)

        return aug1, aug2, aug3  # aug4