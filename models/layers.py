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

# class HPSS(nn.Module):
#     def __init__(self, n_fft=2048, hop_length=512):
#         super(HPSS, self).__init__()
#         self.n_fft = n_fft
#         self.hop_length = hop_length
    
#     def harmonic_filter(self, spectrogram):
#         # Apply a median filter across the time axis (harmonic content is stable over time)
#         harmonic = torch.median(spectrogram, dim=-2, keepdim=True)[0]
#         return harmonic
    
#     def percussive_filter(self, spectrogram):
#         # Apply a median filter across the frequency axis (percussive content is spread across frequencies)
#         percussive = torch.median(spectrogram, dim=-1, keepdim=True)[0]
#         return percussive
    
#     def forward(self, x):
#         # Convertir les samples en spectrogramme
#         window = torch.hann_window(self.n_fft)
#         spectrogram = Faudio.spectrogram(
#             x,
#             n_fft=self.n_fft,
#             hop_length=self.hop_length,
#             pad=0,
#             window=window,
#             win_length=self.n_fft,
#             power=2,
#             normalized=False
#         )
        
#         # Appliquer les filtres harmoniques et percussifs
#         harmonic = self.harmonic_filter(spectrogram)
#         percussive = self.percussive_filter(spectrogram)
        
#         # Composantes résiduelles
#         percussive_res = torch.clamp(spectrogram - harmonic, min=0)
#         harmonic_res = torch.clamp(spectrogram - percussive, min=0)
        
#         # Reconstruire les formes d'onde
#         waveform_harmonic = Faudio.istft(harmonic_res, hop_length=self.hop_length)
#         waveform_percussive = Faudio.istft(percussive_res, hop_length=self.hop_length)
        
#         return waveform_harmonic, waveform_percussive

from typing import Tuple

def softmask(X: torch.Tensor, X_ref: torch.Tensor, power: float = 1, split_zeros: bool = False) -> torch.Tensor:
    if X.shape != X_ref.shape:
        raise ValueError(f'Shape mismatch: {X.shape}!={X_ref.shape}')
    if torch.any(X < 0) or torch.any(X_ref < 0):
        raise ValueError('X and X_ref must be non-negative')
    if power <= 0:
        raise ValueError('power must be strictly positive')

    dtype = X.dtype
    if dtype not in [torch.float16, torch.float32, torch.float64]:
        raise ValueError('data type error')

    Z = torch.max(X, X_ref)
    bad_idx = (Z < torch.finfo(dtype).tiny)
    if bad_idx.sum() > 0: 
        Z[bad_idx] = 1

    if torch.isfinite(torch.tensor(power)):
        mask = (X / Z) ** power
        ref_mask = (X_ref / Z) ** power
        mask /= mask + ref_mask
    else:
        mask = X > X_ref

    return mask


class MedianBlur(nn.Module):
    def __init__(self, kernel_size: Tuple[int, int], channel: int, reduce_method: str = 'median', device: str = 'cpu'):
        super(MedianBlur, self).__init__()
        self.device = device
        self.kernel = self.get_binary_kernel2d(kernel_size).repeat(channel, 1, 1, 1).to(self.device)
        self.padding = self._compute_zero_padding(kernel_size)
        self.reduce_method = reduce_method
        
    @staticmethod
    def get_binary_kernel2d(window_size: Tuple[int, int]) -> torch.Tensor:
        window_range = window_size[0] * window_size[1]
        kernel = torch.zeros(window_range, window_range)
        for i in range(window_range):
            kernel[i, i] += 1.0
        return kernel.view(window_range, 1, window_size[0], window_size[1])

    @staticmethod
    def _compute_zero_padding(kernel_size: Tuple[int, int]) -> Tuple[int, int]:
        return (kernel_size[0] - 1) // 2, (kernel_size[1] - 1) // 2

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if not torch.is_tensor(input):
            raise TypeError(f"Input type is not a torch.Tensor. Got {type(input)}")
        if input.ndim != 4:
            raise ValueError(f"Invalid input shape, we expect BxCxHxW. Got: {input.shape}")

        b, c, h, w = input.shape
        features = F.conv2d(input, self.kernel, padding=self.padding, stride=1, groups=c)
        features = features.view(b, c, -1, h, w)

        if self.reduce_method == 'median':
            res = torch.median(features, dim=2)[0]
        elif self.reduce_method == 'mean':
            res = torch.mean(features, dim=2)
        else:
            raise ValueError(f"Unknown reduce method: {self.reduce_method}")
        return res

class HPSS(nn.Module):
    def __init__(self, kernel_size: int, channel: int = 1, power: float = 2.0, mask: bool = False, margin: float = 1.0, reduce_method: str = 'mean', n_fft: int = 2048, hop_length: int = 512, device: str = 'cpu'):
        super(HPSS, self).__init__()
        self.harm_median_filter = MedianBlur(kernel_size=(1, kernel_size), channel=channel, reduce_method=reduce_method, device=device)
        self.perc_median_filter = MedianBlur(kernel_size=(kernel_size, 1), channel=channel, reduce_method=reduce_method, device=device)
        self.hop_length = hop_length
        self.n_fft = n_fft

    def forward(self, x: torch.Tensor, power: float = 2.0, mask: bool = False, margin: float = 1.0) -> dict:
        window = torch.hann_window(self.n_fft).to(x.device)

        if isinstance(margin, (float, int)):
            margin_harm = margin
            margin_perc = margin
        else:
            margin_harm = margin[0]
            margin_perc = margin[1]

        if margin_harm < 1 or margin_perc < 1:
            raise ValueError("Margins must be >= 1.0.")

        
        D = torch.stft(x.squeeze(1), n_fft=self.n_fft, hop_length=self.hop_length, return_complex=True, window=window)
        # print(D.shape)
        S = torch.abs(D).unsqueeze(1)
        phase = torch.angle(D).unsqueeze(1)

        harm = self.harm_median_filter(S)
        perc = self.perc_median_filter(S)

        split_zeros = (margin_harm == 1 and margin_perc == 1)
        mask_harm = softmask(harm, perc * margin_harm, power=power, split_zeros=split_zeros)
        mask_perc = softmask(perc, harm * margin_perc, power=power, split_zeros=split_zeros)

        # print(mask_harm, mask_perc)

        if mask:
            return mask_harm, mask_perc
        
        harm = S * mask_harm
        perc = S * mask_perc

        complex_harm = harm * torch.exp(1j * phase)
        complex_perc = perc * torch.exp(1j * phase)

        # self.plot_spectra(complex_harm, complex_perc)
        
        # Inversion du spectre
        harm_signal = torch.istft(complex_harm.squeeze(1), n_fft=self.n_fft, hop_length=self.hop_length, window=window).unsqueeze(1)
        perc_signal = torch.istft(complex_perc.squeeze(1), n_fft=self.n_fft, hop_length=self.hop_length, window=window).unsqueeze(1)

        return harm_signal, perc_signal

    # def plot_spectra(self, complex_harm, complex_perc):
    #     # Sélectionner le premier élément du batch
    #     complex_harm_first = complex_harm[0].squeeze().cpu().detach().numpy()  # Si complex_harm est de la forme (n_batch, ...)
    #     complex_perc_first = complex_perc[0].squeeze().cpu().detach().numpy()  # Assurez-vous que la dimension est correcte

    #     # Calculer les magnitudes pour l'affichage
    #     magnitude_harm = np.abs(complex_harm_first)
    #     magnitude_perc = np.abs(complex_perc_first)

    #     # Créer la figure
    #     plt.figure(figsize=(12, 6))

    #     # Afficher le spectre harmonique
    #     plt.subplot(1, 2, 1)
    #     plt.imshow(magnitude_harm, aspect='auto', origin='lower', cmap='viridis')
    #     plt.title('Spectre Harmonique')
    #     plt.xlabel('Temps')
    #     plt.ylabel('Fréquence')

    #     # Afficher le spectre percussif
    #     plt.subplot(1, 2, 2)
    #     plt.imshow(magnitude_perc, aspect='auto', origin='lower', cmap='viridis')
    #     plt.title('Spectre Percussif')
    #     plt.xlabel('Temps')
    #     plt.ylabel('Fréquence')

    #     plt.tight_layout()
    #     plt.show()

class LogMelScale(nn.Module):
    def __init__(self, sample_rate=24000, n_stft=2048, win_length=None, hop_length=512, n_mels=128, f_min=150):
        super(LogMelScale, self).__init__()
        self.sample_rate = sample_rate
        self.n_stft = n_stft
        self.win_length = win_length if win_length is not None else n_stft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.f_min = f_min
        
        self.mel_scale = Taudio.MelScale(
            sample_rate=self.sample_rate,
            n_stft=self.n_stft,
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

# Computer la hpss sans revenir en raw sample
class HPSSMel(nn.Module):
    def __init__(self, kernel_size: int, channel: int = 1, power: float = 2.0, mask: bool = False, margin: float = 1.0, reduce_method: str = 'mean', n_fft: int = 2048, hop_length: int = 512, device: str = 'cpu'):
        super(HPSSMel, self).__init__()
        self.harm_median_filter = MedianBlur(kernel_size=(1, kernel_size), channel=channel, reduce_method=reduce_method, device=device)
        self.perc_median_filter = MedianBlur(kernel_size=(kernel_size, 1), channel=channel, reduce_method=reduce_method, device=device)
        self.hop_length = hop_length
        self.n_fft = n_fft


    def forward(self, x: torch.Tensor, power: float = 2.0, mask: bool = False, margin: float = 1.0) -> dict:
        window = torch.hann_window(self.n_fft).to(x.device)

        if isinstance(margin, (float, int)):
            margin_harm = margin
            margin_perc = margin
        else:
            margin_harm = margin[0]
            margin_perc = margin[1]

        if margin_harm < 1 or margin_perc < 1:
            raise ValueError("Margins must be >= 1.0.")

        
        D = torch.stft(x.squeeze(1), n_fft=self.n_fft, hop_length=self.hop_length, return_complex=True, window=window)
        # print(D.shape)
        S = torch.abs(D).unsqueeze(1)
        # phase = torch.angle(D).unsqueeze(1)

        harm = self.harm_median_filter(S)
        perc = self.perc_median_filter(S)

        split_zeros = (margin_harm == 1 and margin_perc == 1)
        mask_harm = softmask(harm, perc * margin_harm, power=power, split_zeros=split_zeros)
        mask_perc = softmask(perc, harm * margin_perc, power=power, split_zeros=split_zeros)

        # print(mask_harm, mask_perc)

        if mask:
            return mask_harm, mask_perc
        
        harm = S * mask_harm
        perc = S * mask_perc

        return harm, perc