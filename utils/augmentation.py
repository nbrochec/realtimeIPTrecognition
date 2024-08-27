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
from torch_audiomentations import PitchShift, AddColoredNoise, Shift, PolarityInversion, Gain, HighPassFilter, LowPassFilter

'''
Principally using torch_audiomentations because:
1. Augmentations are applied inside the training loop.
2. torch_audiomentations provide cuda support enabling GPU computation.
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

        Parameters
        ----------
        data : torch.Tensor
            The data to which augmentations will be applied.

        Returns
        -------
        torch.Tensor
            The augmented data.
        """
        if self.device:
            data = data.to(self.device)
        # Prepare to accumulate the augmented samples
        augmented_data_list = []

        # Apply each augmentation and store results in a list
        for augmentation in self.augmentations:
            if augmentation == 'pitchshift':
                augmented_data_list.append(self.pitch_shift(data))
            elif augmentation == 'timeshift':
                augmented_data_list.append(self.shift(data))
            elif augmentation == 'addnoise':
                augmented_data_list.append(self.add_noise(data))
            elif augmentation == 'polarityinversion':
                augmented_data_list.append(self.polarity_inversion(data))
            elif augmentation == 'gain':
                augmented_data_list.append(self.gain(data))
            elif augmentation == 'hpf':
                augmented_data_list.append(self.highpassfilter(data))
            elif augmentation == 'lpf':
                augmented_data_list.append(self.lowpassfilter(data))
            elif augmentation == 'all':
                augmented_data_list.append(self.pitch_shift(data))
                augmented_data_list.append(self.shift(data))
                augmented_data_list.append(self.add_noise(data))
                augmented_data_list.append(self.polarity_inversion(data))
                augmented_data_list.append(self.gain(data))
                augmented_data_list.append(self.highpassfilter(data))
                augmented_data_list.append(self.lowpassfilter(data))

        augmented_data = torch.cat(augmented_data_list, dim=-1)
        # print(f'{augmented_data.shape}')

        return augmented_data

    def pitch_shift(self, data):
        """
        Applies pitch shift augmentation to the data.
        """
        transform = PitchShift(min_transpose_semitones=-12.0, max_transpose_semitones=12.0, p=1, sample_rate=self.sr)
        data = transform(data)
        return data

    def shift(self, data):
        """
        Applies time shift augmentation to the data.
        """
        transform=Shift(rollover=True, p=1)
        data = transform(data)
        return data

    def add_noise(self, data):
        """
        Adds noise to the data.
        """
        transform=AddColoredNoise(f_decay=0, p=1)
        data = transform(data)
        return data
    
    def polarity_inversion(data):
        """
        Applies a polarity inversion to the data.
        """
        transform=PolarityInversion(p=1)
        data = transform(data)
        return data
    
    def gain(self, data):
        """
        Multiply the audio by a random amplitude factor to reduce or increase the volume.
        """
        transform=Gain(p=1)
        data = transform(data)
        return data
    
    def highpassfilter(self, data):
        """
        Apply high-pass filtering to the input audio.
        """
        transform=HighPassFilter(p=1)
        data = transform(data)
        return data
    
    def lowpassfilter(self, data):
        """
        Apply low-pass filtering to the input audio.
        """
        transform=LowPassFilter(p=1)
        data = transform(data)
        return data