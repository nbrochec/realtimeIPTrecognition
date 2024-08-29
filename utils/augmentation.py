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
import torch.nn.functional as F
from torch_audiomentations import PitchShift, AddColoredNoise, Shift, PolarityInversion, Gain, HighPassFilter, LowPassFilter

'''
Principally using torch_audiomentations because:
1. Augmentations are applied inside the training loop.
2. torch_audiomentations enables GPU computation to generate augmentated data.
'''

class ApplyAugmentations:
    def __init__(self, augmentations, sample_rate, device=None):
        """
        Initializes the augmentation class with the list of augmentations to apply.

        Parameters
        ----------
        augmentations : list
            List of names of augmentations to apply.
        device : torch.device, optional
            Device to which the data should be moved (default is None).
        """
        self.augmentations = augmentations
        self.sr = sample_rate
        self.device = device

    def apply(self, data):
        """
        Applies the selected augmentations to the given data.
        """
        if self.device:
            data = data.to(self.device)

        original_size = data.size(-1)
        augmented_data_list = []

        for augmentation in self.augmentations:
            if augmentation == 'pitchshift':
                aug_data = self.pitch_shift(data)
            elif augmentation == 'timeshift':
                aug_data = self.shift(data)
            elif augmentation == 'addnoise':
                aug_data = self.add_noise(data)
            elif augmentation == 'polarityinversion':
                aug_data = self.polarity_inversion(data)
            elif augmentation == 'gain':
                aug_data = self.gain(data)
            elif augmentation == 'hpf':
                aug_data = self.highpassfilter(data)
            elif augmentation == 'lpf':
                aug_data = self.lowpassfilter(data)
            elif augmentation == 'all':
                aug_data = [
                    self.pitch_shift(data),
                    self.shift(data),
                    self.add_noise(data),
                    self.polarity_inversion(data),
                    self.gain(data),
                    self.highpassfilter(data),
                    self.lowpassfilter(data)
                ]
                augmented_data_list.extend(aug_data)
                continue 

            aug_data = self.pad_or_trim(aug_data, original_size)
            augmented_data_list.append(aug_data)

        augmented_data = torch.cat(augmented_data_list, dim=0)

        return augmented_data

    def pad_or_trim(self, data, original_size):
        current_size = data.size(-1)
        
        if current_size > original_size:
            data = data[..., :original_size]
        elif current_size < original_size:
            padding = (0, original_size - current_size)
            data = F.pad(data, pad=padding, mode='constant', value=0)
        
        return data

    def pitch_shift(self, data):
        transform = PitchShift(min_transpose_semitones=-12.0, max_transpose_semitones=12.0, p=1, sample_rate=self.sr)
        return transform(data)

    def shift(self, data):
        transform = Shift(rollover=True, p=1)
        return transform(data)

    def add_noise(self, data):
        transform = AddColoredNoise(p=1)
        return transform(data, sample_rate=self.sr)
    
    def polarity_inversion(self, data):
        transform = PolarityInversion(p=1)
        return transform(data)
    
    def gain(self, data):
        transform = Gain(p=1)
        return transform(data)
    
    def highpassfilter(self, data):
        transform = HighPassFilter(p=1)
        return transform(data, sample_rate=self.sr)
    
    def lowpassfilter(self, data):
        transform = LowPassFilter(p=1)
        return transform(data, sample_rate=self.sr)
    
    def get_aug_nbr(self):
        aug_nbr = len(self.augmentations)
        if self.augmentations == ['all']:
            aug_nbr = 7
        return aug_nbr