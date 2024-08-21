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
import torch_audiomentations

'''
Principally using torch_audiomentations because:
1. Augmentations are generated inside the training loop.
2. torch_audiomentations provide cuda support enabling GPU computation.
'''

class ApplyAugmentations:
    def __init__(self, augmentations, device=None):
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
        if 'pitchshift' in self.augmentations:
            data = self.pitch_shift(data)
        if 'timeshift' in self.augmentations:
            data = self.time_shift(data)
        if 'addnoise' in self.augmentations:
            data = self.add_noise(data)
        # Apply more augmentations as needed
        return data

    def pitch_shift(self, data):
        """
        Applies pitch shift augmentation to the data.

        Parameters
        ----------
        data : torch.Tensor
            The data to augment.

        Returns
        -------
        torch.Tensor
            The pitch-shifted data.
        """

        return data

    def time_shift(self, data):
        """
        Applies time shift augmentation to the data.

        Parameters
        ----------
        data : torch.Tensor
            The data to augment.

        Returns
        -------
        torch.Tensor
            The time-shifted data.
        """

        return data

    def add_noise(self, data):
        """
        Adds noise to the data.

        Parameters
        ----------
        data : torch.Tensor
            The data to augment.

        Returns
        -------
        torch.Tensor
            The data with added noise.
        """

        return data