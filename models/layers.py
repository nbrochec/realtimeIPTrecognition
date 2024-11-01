#############################################################################
# layers.py 
# Nicolas Brochec
# TOKYO UNIVERSITY OF THE ARTS
# 東京藝術大学音楽音響創造科
# ERC Reach
# GNU General Public License v3.0
#############################################################################
# Code description:
# Implement different custom layers
#############################################################################

import torch
import torch.nn as nn
import torchaudio.transforms as Taudio

class LogMelSpectrogramLayer(nn.Module):
    def __init__(self, sample_rate=24000, n_fft=2048, win_length=None, hop_length=512, n_mels=128, f_min=150, f_max=None, center=True):
        super(LogMelSpectrogramLayer, self).__init__()
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.win_length = win_length if win_length is not None else n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.f_min = f_min
        self.f_max = f_max
        self.center = center
        
        self.mel_scale = Taudio.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=self.n_fft,
            win_length=self.win_length,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
            f_min=self.f_min,
            f_max=self.f_max,
            center=self.center,
        )

        self.amplitude_to_db = Taudio.AmplitudeToDB(stype='power')

    def min_max_normalize(self, tensor: torch.Tensor, min_val: float = 0.0, max_val: float = 1.0) -> torch.Tensor:
        """Normalizes the input tensor to the specified range [min_val, max_val]."""
        min_tensor = torch.as_tensor(min_val, dtype=tensor.dtype, device=tensor.device)
        max_tensor = torch.as_tensor(max_val, dtype=tensor.dtype, device=tensor.device)
        eps = 1e-9
        
        t_min = tensor.min()
        t_max = tensor.max()

        t_range = t_max - t_min + eps
        normalized_tensor = (tensor - t_min) / t_range
        
        scaled_tensor = normalized_tensor * (max_tensor - min_tensor) + min_tensor
        return scaled_tensor

    def forward(self, x):
        x = self.mel_scale(x)
        x = self.amplitude_to_db(x)
        x = self.min_max_normalize(x)
        return x.to(torch.float32)

class customCNN2D(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size, padding):
        super(customCNN2D, self).__init__()
        self.conv = nn.Conv2d(in_channels=input_channels, out_channels=output_channels, kernel_size=kernel_size, stride=1, padding=padding)
        self.batch = nn.BatchNorm2d(output_channels)
        self.activ = nn.LeakyReLU(negative_slope=0.01)

    def forward(self, x):
        x = self.conv(x)
        x = self.batch(x)
        x = self.activ(x)
        return x
