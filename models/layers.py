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
    
class spectralEnergyExtractor(nn.Module):
    def __init__(self):
        super(spectralEnergyExtractor, self).__init__()

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
        rms = torch.sqrt(torch.mean(x ** 2, dim=2))
        x = self.min_max_normalize(rms)
        return x
    
class EnvelopeFollowingLayerTorchScript(nn.Module):
    def __init__(self, n_fft=1024, hop_length=512, smoothing_factor=None):
        super(EnvelopeFollowingLayerTorchScript, self).__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.smoothing_factor = smoothing_factor

    def forward(self, x):
        batch_size, n_channels, time = x.shape
        window = torch.hann_window(self.n_fft).to(x.device)
        envelope_list = []
        for i in range(n_channels):
            stft_result = torch.stft(x[:, i, :], n_fft=self.n_fft, hop_length=self.hop_length, window=window, return_complex=True)
            
            # Obtenir le signal analytique en annulant les parties négatives du spectre de fréquence
            stft_analytic = stft_result.clone()
            stft_analytic[..., self.n_fft//2+1:] = 0  # Supprimer les fréquences négatives
            
            # Effectuer l'iSTFT (inverse STFT) pour revenir dans le domaine temporel
            analytic_signal = torch.istft(stft_analytic, n_fft=self.n_fft, hop_length=self.hop_length, window=window, return_complex=False)
            
            # Prendre le module du signal analytique pour obtenir l'enveloppe
            envelope = torch.abs(analytic_signal)
            envelope_list.append(envelope.unsqueeze(1))
        
        # Concatenation des enveloppes des différents canaux
        envelope_output = torch.cat(envelope_list, dim=1)  # Shape: [batch, n_channels, time]

        # Lissage optionnel (par exemple, en appliquant une moyenne glissante ou un filtre passe-bas)
        if self.smoothing_factor is not None:
            envelope_output = F.avg_pool1d(envelope_output, kernel_size=self.smoothing_factor, stride=1, padding=self.smoothing_factor//2)

        return envelope_output

class EnvelopeMelLayer(nn.Module):
    def __init__(self, sr):
        super(EnvelopeMelLayer, self).__init__()  # Appel au constructeur de nn.Module
        self.sr = sr
    
    def forward(self, mel_spec, envelope):
        """
        mel_spec: Input mel spectrogram (n_batch, 1, n_mels, time_frames)
        envelope: Envelope signal with dimensions (n_batch, n_samples) or similar to the audio length
        """
        
        # Step 2: Ensure the envelope is compatible with the time frames in the mel spectrogram
        # Reshape envelope: from (n_batch, n_samples) -> (n_batch, 1, 1, time_frames)
        envelope = F.interpolate(envelope.unsqueeze(1).unsqueeze(1), size=mel_spec.shape[-1], mode='linear')
        
        # Step 3: Apply the envelope to each frame (multiply along the time axis)
        mel_spec_enveloped = mel_spec * envelope  # Apply envelope to mel-spectrogram

        return mel_spec_enveloped