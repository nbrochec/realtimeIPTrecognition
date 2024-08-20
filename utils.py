#############################################################################
# utils.py 
# Nicolas Brochec
# TOKYO UNIVERSITY OF THE ARTS
# 東京藝術大学音楽音響創造科
# ERC Reach
# MIT license 2024
#############################################################################
# Code description:
# Utilities defition
#############################################################################

from os.path import join, dirname, basename, abspath, normpath, isdir
from os import listdir
from glob import glob
import torchaudio.transforms as Taudio
import torch.nn.functional as Fnn
import torch

class customLogMelSpectrogram():
    def __init__(self, sample_rate=24000, n_fft=2048, win_length=None, hop_length=512, n_mels=128, f_min=150, f_max=None):
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
    
    def __call__(self, waveform):
        mel_spec = self.mel_scale(waveform)
        mel_spec_db = self.amplitude_to_db(mel_spec)
        mel_spec_db = Fnn.normalize(mel_spec_db, dim=2)
        mel_spec_db = Fnn.normalize(mel_spec_db, dim=3)
        return mel_spec_db
    
    def process_segment(self, waveform, segment_length=7680):
        """Divides the waveform into segments of a fixed length and applies the Mel transformation."""
        segments = []
        for start in range(0, waveform.size(1) - segment_length + 1, segment_length):
            segment = waveform[:, start:start + segment_length]
            mel_spec_db = self.__call__(segment)
            segments.append(mel_spec_db)
        return torch.stack(segments, dim=0)  # (num_segments, 1, 128, 15)




