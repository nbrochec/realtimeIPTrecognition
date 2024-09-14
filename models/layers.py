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
import torch.nn.functional as F
import torchaudio.functional as Faudio

class LogMelSpectrogramLayer(nn.Module):
    def __init__(self, sample_rate=24000, n_fft=2048, win_length=None, hop_length=512, n_mels=128, f_min=150, f_max=None):
        super(LogMelSpectrogramLayer, self).__init__()
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

    def min_max_normalize(self, t, min=0, max=1):
        min = 0
        max = 1
        if ((torch.max(t)-torch.min(t)) == 0):
            upsilon = 0.00001
            t_std = (t - torch.min(t)) / ((torch.max(t)-torch.min(t))+upsilon)
            t_scaled = t_std * (max - min) + min
        else:
            t_std = (t - torch.min(t)) / (torch.max(t)-torch.min(t))
            t_scaled = t_std * (max - min) + min
        return t_scaled

    def forward(self, x):
        x = self.mel_scale(x)
        x = self.amplitude_to_db(x)
        x = self.min_max_normalize(x)
        return x

class EnvelopeExtractor(nn.Module):
    def __init__(self, sample_rate=24000, cutoff_freq=10, Q=0.707):
        super(EnvelopeExtractor, self).__init__()
        self.sr = sample_rate
        self.cutoff = cutoff_freq
        self.Q = Q

    def forward(self, x):
        x = torch.abs(x)
        x = Faudio.highpass_biquad(x, sample_rate=self.sr, cutoff_freq=self.cutoff, Q=self.Q)
        return x

class custom2DCNN(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size, padding):
        super(custom2DCNN, self).__init__()
        self.conv = nn.Conv2d(in_channels=input_channels, out_channels=output_channels, kernel_size=kernel_size, stride=1, padding=padding)
        self.batch = nn.BatchNorm2d(output_channels)
        self.activ = nn.LeakyReLU()
        # self.drop = nn.Dropout2d(0.25)

    def forward(self, x):
        x = self.conv(x)
        x = self.batch(x)
        x = self.activ(x)
        # x = self.drop(x)
        return x

class custom1DCNN(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size, padding, dilation):
        super(custom1DCNN, self).__init__()
        self.conv = nn.Conv1d(in_channels=input_channels, out_channels=output_channels,kernel_size=kernel_size, padding=padding, dilation=dilation)
        self.batch = nn.BatchNorm1d(output_channels)
        self.activ = nn.GELU()
        self.avg = nn.AvgPool1d(4)
    
    def forward(self,x):
        x = self.conv(x)
        x = self.batch(x)
        x = self.activ(x)
        # x = self.drop(x)
        return x