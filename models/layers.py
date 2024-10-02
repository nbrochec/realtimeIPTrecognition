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

class EnvelopeExtractor(nn.Module):
    def __init__(self, sample_rate=24000, cutoff_freq=10, Q=0.707):
        super(EnvelopeExtractor, self).__init__()
        self.sr = sample_rate
        self.cutoff = cutoff_freq
        self.Q = Q

    def forward(self, x):
        x = torch.abs(x)
        x = Faudio.highpass_biquad(x, sample_rate=self.sr, cutoff_freq=float(self.cutoff), Q=self.Q)
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
            envelope = self.rms_normalize(envelope)
            envelope_list.append(envelope.unsqueeze(1))

        envelope_output = torch.cat(envelope_list, dim=1)

        envelope_output = F.avg_pool1d(envelope_output, kernel_size=self.smoothing_factor, stride=1, padding=self.smoothing_factor//2)

        # envelope = envelope_output[0, 0, :].cpu().numpy()

        # plt.plot(envelope)
        # plt.title('Envelope Output')
        # plt.xlabel('Time')
        # plt.ylabel('Amplitude')
        # plt.show()

        return envelope_output