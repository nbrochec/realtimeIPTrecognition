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

import torch
from audiomentations import PitchShift, AddColorNoise, Shift, PolarityInversion, Gain, HighPassFilter, Trim
from audiomentations import LowPassFilter, Mp3Compression, ClippingDistortion, BitCrush, AirAbsorption, Aliasing
import torch.nn.functional as F
import numpy as np
import librosa, random

class AudioOnlineTransforms:
    def __init__(self, args):
        self.sr = args.sr
        self.device = args.device
        self.online_augment = args.online_augment.split()

    def tensor_to_array(self, data):
        data_numpy = data.cpu().squeeze(1).detach().numpy()
        return data_numpy
    
    def shift(self, data):
        transform = Shift(rollover=True, p=1)
        return transform(data, sample_rate= self.sr)
    
    def pitch_shift(self, data):
        transform = PitchShift(min_semitones=-12.0, max_semitones=12.0, p=1)
        return transform(data, sample_rate= self.sr)
    
    def add_noise(self, data):
        transform = AddColorNoise(p=1)
        return transform(data, sample_rate=self.sr)
    
    def polarity_inversion(self, data):
        transform = PolarityInversion(p=1)
        return transform(data, sample_rate= self.sr)
    
    def gain(self, data):
        transform = Gain(p=1)
        return transform(data, sample_rate= self.sr)
    
    def highpassfilter(self, data):
        transform = HighPassFilter(p=1)
        return transform(data, sample_rate=self.sr)
    
    def lowpassfilter(self, data):
        transform = LowPassFilter(p=1)
        return transform(data, sample_rate=self.sr)
    
    def clippingdisto(self, data):
        transform = ClippingDistortion(p=1)
        return transform(data, sample_rate=self.sr)
    
    def bitcrush(self, data):
        transform = BitCrush(p=1)
        return transform(data, sample_rate=self.sr)
    
    def airabso(self, data):
        transform = AirAbsorption(p=1)
        return transform(data, sample_rate=self.sr)
    
    def aliasing(self, data):
        transform = Aliasing(min_sample_rate=8000, max_sample_rate=self.sr//2, p=1)
        return transform(data, sample_rate=self.sr)
    
    def mp3comp(self, data):
        transform = Mp3Compression(p=1)
        return transform(data, sample_rate=self.sr)
    
    def trim(self, data):
        transform = Trim(top_db=30.0,p=1)
        return transform(data, sample_rate=self.sr)

    def pad_or_trim(self, data, original_size):
        current_size = data.shape[1]
        if current_size > original_size:
            data = data[..., :original_size]
        elif current_size < original_size:
            padding = (0, original_size - current_size)
            data = np.pad(data, pad_width=padding, mode='constant', constant_values=0)
        return data
    def none(self, data):
        return data
    
    def __call__(self, data):
        data_numpy = self.tensor_to_array(data)
        original_size = data_numpy.shape[1]
        aug_data_list = []

        aug_dict = {
            'none': self.none,
            'pitchshift': self.pitch_shift,
            'timeshift': self.shift,
            'polarityinversion': self.polarity_inversion,
            'hpf': self.highpassfilter,
            'lpf': self.lowpassfilter,
            'clipping': self.clippingdisto,
            'bitcrush': self.bitcrush,
            'airabso': self.airabso,
            'aliasing': self.aliasing,
            'mp3comp': self.mp3comp,
            'trim': self.trim
        }

        for augmentation in self.online_augment:
            if augmentation in aug_dict and np.random.rand() < 0.5:
                if augmentation != 'none':
                    data_numpy = aug_dict[augmentation](data_numpy)
        
        aug_data = self.pad_or_trim(data_numpy, original_size)
        aug_data_list.append(aug_data)

        aug_data_tensor = torch.tensor(aug_data).unsqueeze(1).to(self.device).to(torch.float32)

        return aug_data_tensor
    
class AudioOfflineTransforms:
    def __init__(self, args):
        self.sr = args.sr
        self.device = args.device

    def tensor_to_array(self, data):
        data_numpy = data.cpu().squeeze(1).detach().numpy()
        return data_numpy

    def custom_detune(self, data):
        random_tuning= np.random.uniform(-100, 100) + 440
        change_tuning = librosa.A4_to_tuning(random_tuning)
        data = librosa.effects.pitch_shift(data, sr=self.sr, n_steps=change_tuning, bins_per_octave=12)
        return data

    def custom_gaussnoise(self, data):
        n_channels, length = data.shape
        data = data + np.random.normal(0, 0.01, size=(n_channels, length)) * np.random.randn(n_channels, length)
        data = data[:, :length]
        return data
    
    def pad_or_trim(self, data, original_size):
        current_size = data.shape[1]
        if current_size > original_size:
            data = data[..., :original_size]
        elif current_size < original_size:
            padding = (0, original_size - current_size)
            data = np.pad(data, pad_width=padding, mode='constant', constant_values=0)
        return data
    
    def __call__(self, data):
        data_numpy = self.tensor_to_array(data)
        original_size = data_numpy.shape[1]

        aug_dict = {
            'detune': self.custom_detune,
            'gaussnoise': self.custom_gaussnoise,
        }

        detuned = aug_dict['detune'](data_numpy)
        noised = aug_dict['gaussnoise'](data_numpy)

        aug_detuned = self.pad_or_trim(detuned, original_size)
        aug_noised = self.pad_or_trim(noised, original_size)

        aug1 = torch.tensor(aug_detuned).to(torch.float32)
        aug2 = torch.tensor(aug_noised).to(torch.float32)

        return aug1, aug2