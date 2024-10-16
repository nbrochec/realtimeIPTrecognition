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

import numpy as np

from typing import Tuple, Union, Dict

import matplotlib.pyplot as plt

class LogMelSpectrogramLayer(nn.Module):
    def __init__(self, sample_rate=24000, n_fft=2048, win_length=None, hop_length=512, n_mels=128, f_min=150):
        super(LogMelSpectrogramLayer, self).__init__()
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.win_length = win_length if win_length is not None else n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.f_min = f_min
        
        self.mel_scale = Taudio.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=self.n_fft,
            win_length=self.win_length,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
            f_min=self.f_min
        )

        self.amplitude_to_db = Taudio.AmplitudeToDB(stype='power')

    def min_max_normalize(self, t: torch.Tensor, min: float = 0.0, max: float = 1.0) -> torch.Tensor:
        min_tensor = torch.tensor(min, dtype=t.dtype, device=t.device)
        max_tensor = torch.tensor(max, dtype=t.dtype, device=t.device)
        eps = 1e-5
        t_min = torch.min(t)
        t_max = torch.max(t)

        if (t_max - t_min) == 0:
            t_std = (t - t_min) / ((t_max - t_min) + eps)
        else:
            t_std = (t - t_min) / (t_max - t_min)
        
        t_scaled = t_std * (max_tensor - min_tensor) + min_tensor
        return t_scaled

    def forward(self, x):
        x = self.mel_scale(x)
        x = self.amplitude_to_db(x)
        # x = torch.where(torch.isinf(x), torch.tensor(0.0).to(x.device), x)
        x = self.min_max_normalize(x)
        return x.to(torch.float32)
    
class LogMelSpectrogramLayerERANN(nn.Module):
    def __init__(self, sample_rate=24000, n_fft=2048, win_length=None, hop_length=512, n_mels=128, f_min=150):
        super(LogMelSpectrogramLayerERANN, self).__init__()
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.win_length = win_length if win_length is not None else n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.f_min = f_min
        
        self.mel_scale = Taudio.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=self.n_fft,
            win_length=self.win_length,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
            f_min=self.f_min
        )

        self.amplitude_to_db = Taudio.AmplitudeToDB(stype='power')

    def forward(self, x):
        x = self.mel_scale(x)
        x = self.amplitude_to_db(x)
        return x.to(torch.float32)


class custom2DCNN(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size, padding):
        super(custom2DCNN, self).__init__()
        self.conv = nn.Conv2d(in_channels=input_channels, out_channels=output_channels, kernel_size=kernel_size, stride=1, padding=padding)
        self.batch = nn.BatchNorm2d(output_channels)
        self.activ = nn.LeakyReLU(negative_slope=0.01)
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
        # self.avg = nn.AvgPool1d(4)
    
    def forward(self,x):
        x = self.conv(x)
        x = self.batch(x)
        x = self.activ(x)
        # x = self.drop(x)
        return x
    
class EnvelopeFollowingLayerTorchScript(nn.Module):
    def __init__(self, n_fft=1024, hop_length=512, smoothing_factor=None):
        super(EnvelopeFollowingLayerTorchScript, self).__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.smoothing_factor = smoothing_factor
        self.eps = 1e-8
        self.n_channels = 1

    def min_max_normalize(self, t: torch.Tensor, min: float = 0.0, max: float = 1.0) -> torch.Tensor:
        min_tensor = torch.tensor(min, dtype=t.dtype, device=t.device)
        max_tensor = torch.tensor(max, dtype=t.dtype, device=t.device)
        t_min = torch.min(t)
        t_max = torch.max(t)

        if (t_max - t_min) == 0:
            t_std = (t - t_min) / ((t_max - t_min) + self.eps)
        else:
            t_std = (t - t_min) / (t_max - t_min)
        
        t_scaled = t_std * (max_tensor - min_tensor) + min_tensor
        return t_scaled
    
    def rms_normalize(self, signal):
        rms = torch.sqrt(torch.mean(signal ** 2, dim=-1, keepdim=True))
        signal = signal / (rms + self.eps)
        return signal

    def forward(self, x):
        window = torch.hann_window(self.n_fft).to(x.device)
        envelope_list = []

        for i in range(self.n_channels):
            stft_result = torch.stft(x[:, i, :], n_fft=self.n_fft, hop_length=self.hop_length, window=window, return_complex=True)
            
            stft_analytic = stft_result.clone()
            stft_analytic[..., self.n_fft//2+1:] = 0 
            
            analytic_signal = torch.istft(stft_analytic, n_fft=self.n_fft, hop_length=self.hop_length, window=window, return_complex=False)
            
            envelope = torch.abs(analytic_signal)
            # envelope = self.rms_normalize(envelope)
            # different normalisation
            envelope = envelope / envelope.amax(dim=-1, keepdim=True)
            envelope_list.append(envelope.unsqueeze(1))

        envelope_output = torch.cat(envelope_list, dim=1)

        if self.smoothing_factor is not None:
            # Enhanced Smoothing
            kernel = torch.ones(1, 1, self.smoothing_factor).to(x.device) / self.smoothing_factor
            envelope_output = F.pad(envelope_output, (self.smoothing_factor//2, self.smoothing_factor//2), mode='reflect')
            envelope_output = F.conv1d(envelope_output, kernel, stride=1)

        # envelope_output = F.avg_pool1d(envelope_output, kernel_size=self.smoothing_factor, stride=1, padding=self.smoothing_factor//2)

        # envelope = envelope_output[0, 0, :].cpu().numpy()

        # plt.plot(envelope)
        # plt.title('Envelope Output')
        # plt.xlabel('Time')
        # plt.ylabel('Amplitude')
        # plt.show()

        return envelope_output

class ARB(nn.Module):
    def __init__(self, cin, cout, x, y):
        super(ARB, self).__init__()
        
        K1, K2, P = self.get_kernels_and_padding(x)
        
        self.use_residual = (cin == cout and x == y == 1)

        self.batchnorm1 = nn.BatchNorm2d(cin)
        self.leaky_relu1 = nn.LeakyReLU(negative_slope=0.01)

        self.conv1 = nn.Conv2d(cin, cout, kernel_size=K1, stride=(x, y), padding=P)
        self.batchnorm2 = nn.BatchNorm2d(cout)
        self.leaky_relu2 = nn.LeakyReLU(negative_slope=0.01)

        self.conv2 = nn.Conv2d(cout, cout, kernel_size=K2, stride=(1, 1), padding=P)
        
        if not self.use_residual:
            self.conv_res = nn.Conv2d(cin, cout, kernel_size=(1, 1), stride=(x, y), padding=(0, 0))

    def get_kernels_and_padding(self, z):
        if z == 1 or z == 2:
            K1 = K2 = (3, 3)
            P = (1, 1)
        elif z == 4:
            K1 = K2 = (6, 5)
            P = (2, 2)
        else:
            raise ValueError("Unsupported stride size")
        return K1, K2, P

    def forward(self, x):
        identity = x
        
        out = self.batchnorm1(x)
        out = self.leaky_relu1(out)
        out = self.conv1(out)
        out = self.batchnorm2(out)
        out = self.leaky_relu2(out)
        out = self.conv2(out)
        
        if not self.use_residual:
            identity = self.conv_res(identity)
        
        out += identity
        return out
    


class ARB1d(nn.Module):
    def __init__(self, cin, cout, k, d):
        super(ARB1d, self).__init__()
        
        self.use_residual = (cin == cout)

        P = (k - 1) // 2
        self.conv1 = nn.Conv1d(cin, cout, kernel_size=k, padding=P, dilation=d)
        self.batchnorm = nn.BatchNorm1d(cout)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.01)

        if not self.use_residual:
            self.conv_res = nn.Conv1d(cin, cout, kernel_size=1, padding=0)

    def forward(self, x):
        identity = x
        
        out = self.batchnorm(self.conv1(x))
        out = self.leaky_relu(out)
        
        if not self.use_residual:
            identity = self.conv_res(identity)

        if out.shape != identity.shape:
            min_length = min(out.shape[2], identity.shape[2])
            out = out[:, :, :min_length]
            identity = identity[:, :, :min_length]

        out += identity
        return out



class customARB1D(nn.Module):
    def __init__(self, cin, cout, k, dilation=1, avgpool=None, dropout=0.25, first=False):
        super(customARB1D, self).__init__()

        self.use_residual = (cin == cout)

        self.conv2d_1 = nn.Conv1d(cin, cout, kernel_size=k, padding="same", dilation=dilation)

        if not self.use_residual:
            self.conv_res = nn.Conv1d(cin, cout, kernel_size=1, padding="same", dilation=dilation)
        else:
            self.conv_res = nn.Identity()

        self.gelu = nn.GELU()
        self.batchnorm = nn.BatchNorm1d(cout)
        self.batchnorm_in = nn.BatchNorm1d(cin)

        self.dropout = nn.Dropout1d(dropout)

        self.first = first

        self.avgpool = nn.AvgPool1d(avgpool) if avgpool is not None else None

    def forward(self, x):
        if self.first is False:
            x = self.batchnorm_in(x)
            x = self.gelu(x)

        res = x

        # Conv, batchnorm et activation
        x = self.conv2d_1(x)
        x = self.batchnorm(x)
        x = self.gelu(x)

        # Résidu
        res = self.conv_res(res)

        # Ajouter le chemin résiduel
        x += res

        if self.avgpool is not None:
            x = self.avgpool(x)

        # Appliquer le dropout
        out = self.dropout(x)
        return out


class customARB(nn.Module):
    def __init__(self, cin, cout, k, maxpool=None, dropout=0.25, first=False):
        super(customARB, self).__init__()

        self.use_residual = (cin == cout)

        self.conv2d_1 = nn.Conv2d(cin, cout, kernel_size=k, padding="same")

        if not self.use_residual:
            self.conv_res = nn.Conv2d(cin, cout, kernel_size=1, padding="same")
        else:
            self.conv_res = nn.Identity()

        self.first = first

        self.leaky_relu = nn.LeakyReLU(negative_slope=0.01)
        self.batchnorm = nn.BatchNorm2d(cout)
        self.batchnorm_in = nn.BatchNorm2d(cin)

        self.dropout = nn.Dropout2d(dropout)

        self.maxpool = nn.MaxPool2d(maxpool) if maxpool is not None else None

    def forward(self, x):

        if self.first is False:
            x = self.batchnorm_in(x)
            x = self.leaky_relu(x)

        res = x

        # Conv, batchnorm et activation
        x = self.conv2d_1(x)
        x = self.batchnorm(x)
        x = self.leaky_relu(x)

        # Résidu
        res = self.conv_res(res)

        # Ajouter le chemin résiduel
        x += res

        # Appliquer maxpool seulement si nécessaire
        if self.maxpool is not None:
            x = self.maxpool(x)

        # Appliquer le dropout
        out = self.dropout(x)
        return out