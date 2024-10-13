#############################################################################
# models.py 
# Nicolas Brochec
# TOKYO UNIVERSITY OF THE ARTS
# 東京藝術大学音楽音響創造科
# ERC Reach
# GNU General Public License v3.0
#############################################################################
# Code description:
# Implement different model architectures
#############################################################################

import torch
import torch.nn as nn
import torchaudio.transforms as T
import torch.nn.functional as F
import torchaudio.functional as Faudio

from models.layers import LogMelSpectrogramLayer, custom2DCNN, custom1DCNN, EnvelopeFollowingLayerTorchScript, HPSS, LogMelScale, HPSSMel
from utils.constants import SEGMENT_LENGTH

class v1(nn.Module):
    def __init__(self, output_nbr, args):
        super(v1, self).__init__()

        self.sr = args.sr

        self.logmel = LogMelSpectrogramLayer(sample_rate=self.sr)
        
        self.cnn = nn.Sequential(
            custom2DCNN(1, 40, (2,3), "same"),
            custom2DCNN(40, 40, (2,3), "same"),
            nn.MaxPool2d((2, 1)),
            nn.Dropout2d(0.25),
            custom2DCNN(40, 80, (2,3), "same"),
            custom2DCNN(80, 80, (2,3), "same"),
            nn.MaxPool2d((2, 3)),
            nn.Dropout2d(0.25),
            custom2DCNN(80, 160, 2, "same"),
            nn.MaxPool2d((2, 1)),
            nn.Dropout2d(0.25),
            custom2DCNN(160, 160, 2, "same"),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.25),
            custom2DCNN(160, 160, 2, "same"),
            nn.MaxPool2d((2, 1)),
            nn.Dropout2d(0.25),
            custom2DCNN(160, 160, 2, "same"),
            nn.MaxPool2d((4, 2)),
            nn.Dropout2d(0.25),
        )

        self.fc = nn.Sequential(
            nn.Linear(160, 80),
            nn.Linear(80, 40),
            nn.Linear(40, output_nbr)
        )

    @torch.jit.export
    def get_attributes(self):
        return self.sr

    def forward(self, x):
        x = self.logmel(x)
        x = self.cnn(x)
        x_flat = x.view(x.size(0), -1)
        z = self.fc(x_flat)
        return z

class v2(nn.Module):
    def __init__(self, output_nbr, sr):
        super(v2, self).__init__()

        self.logmel = LogMelSpectrogramLayer(sample_rate=sr)
        
        self.cnn = nn.Sequential(
            custom2DCNN(1, 64, (2,3), "same"),
            custom2DCNN(64, 64, (2,3), "same"),
            nn.MaxPool2d((2, 1)),
            nn.Dropout2d(0.25),
            custom2DCNN(64, 128, (2,3), "same"),
            custom2DCNN(128, 128, (2,3), "same"),
            nn.MaxPool2d((2, 3)),
            nn.Dropout2d(0.25),
            custom2DCNN(128, 256, 2, "same"),
            nn.MaxPool2d((2, 1)),
            nn.Dropout2d(0.25),
            custom2DCNN(256, 256, 2, "same"),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.25),
            custom2DCNN(256, 512, 2, "same"),
            nn.MaxPool2d((2, 1)),
            nn.Dropout2d(0.25),
            custom2DCNN(512, 512, 2, "same"),
            nn.MaxPool2d((2, 1)),
            nn.Dropout2d(0.25),
            custom2DCNN(512, 512, 2, "same"),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.25),
        )

        self.fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.ELU(),
            nn.Linear(256, 128),
            nn.ELU(),
            nn.Linear(128, output_nbr),
        )

    def forward(self, x):
        x = self.logmel(x)
        x = self.cnn(x)
        x_flat = x.view(x.size(0), -1)
        z = self.fc(x_flat)
        return z
    
class v3(nn.Module):
    def __init__(self, output_nbr, sr):
        super(v3, self).__init__()

        self.logmel = LogMelSpectrogramLayer(sample_rate=sr)
        
        self.cnn = nn.Sequential(
            custom2DCNN(1, 64, (2,3), "same"),
            custom2DCNN(64, 64, (2,3), "same"),
            nn.MaxPool2d((2, 1)),
            nn.Dropout2d(0.25),
            custom2DCNN(64, 128, (2,3), "same"),
            custom2DCNN(128, 128, (2,3), "same"),
            nn.MaxPool2d((2, 3)),
            nn.Dropout2d(0.25),
            custom2DCNN(128, 256, 2, "same"),
            custom2DCNN(256, 256, 2, "same"),
            nn.MaxPool2d((2, 1)),
            nn.Dropout2d(0.25),
            custom2DCNN(256, 256, 2, "same"),
            custom2DCNN(256, 256, 2, "same"),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.25),
            custom2DCNN(256, 512, 2, "same"),
            custom2DCNN(512, 512, 2, "same"),
            nn.MaxPool2d((2, 1)),
            nn.Dropout2d(0.25),
            custom2DCNN(512, 512, 2, "same"),
            custom2DCNN(512, 512, 2, "same"),
            nn.MaxPool2d((2, 1)),
            nn.Dropout2d(0.25),
            custom2DCNN(512, 512, 2, "same"),
            custom2DCNN(512, 512, 2, "same"),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.25),
        )

        self.fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.ELU(),
            nn.Linear(256, 128),
            nn.ELU(),
            nn.Linear(128, output_nbr),
        )

    def forward(self, x):
        x = self.logmel(x)
        x = self.cnn(x)
        x_flat = x.view(x.size(0), -1)
        z = self.fc(x_flat)
        return z

class v2_1d(nn.Module):
    def __init__(self, output_nbr, sr):
        super(v2_1d, self).__init__()
        self.logmel = LogMelSpectrogramLayer(sample_rate=sr)

        self.env = EnvelopeFollowingLayerTorchScript(sample_rate=sr)

        self.cnn1d = nn.Sequential(
            custom1DCNN(1, 64, 8, "same", 2),
            custom1DCNN(64, 64, 7, "same", 2),
            nn.AvgPool1d(8),
            custom1DCNN(64, 64, 6, "same", 1),
            custom1DCNN(64, 64, 5, "same", 1),
            nn.AvgPool1d(4),
            custom1DCNN(64, 64, 4, "same", 1),
            nn.AvgPool1d(4),
            custom1DCNN(64, 64, 3, "same", 1),
            nn.AvgPool1d(2),
            custom1DCNN(64, 128, 2, "same", 1),
            nn.AvgPool1d(2),
            custom1DCNN(128, 128, 2, "same", 1),
            nn.AvgPool1d(2),
            custom1DCNN(128, 128, 2, "same", 1),
            nn.AvgPool1d(7),
            nn.Dropout1d(0.25),
        )

        self.cnn2d = nn.Sequential(
            custom2DCNN(1, 64, (2,3), "same"),
            custom2DCNN(64, 64, (2,3), "same"),
            nn.MaxPool2d((2, 1)),
            nn.Dropout2d(0.25),
            custom2DCNN(64, 128, (2,3), "same"),
            custom2DCNN(128, 128, (2,3), "same"),
            nn.MaxPool2d((2, 3)),
            nn.Dropout2d(0.25),
            custom2DCNN(128, 256, 2, "same"),
            nn.MaxPool2d((2, 1)),
            nn.Dropout2d(0.25),
            custom2DCNN(256, 256, 2, "same"),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.25),
            custom2DCNN(256, 512, 2, "same"),
            nn.MaxPool2d((2, 1)),
            nn.Dropout2d(0.25),
            custom2DCNN(512, 512, 2, "same"),
            nn.MaxPool2d((2, 1)),
            nn.Dropout2d(0.25),
            custom2DCNN(512, 512, 2, "same"),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.25),
        )

        self.fc = nn.Sequential(
            nn.Linear(512 + 128, 128),
            nn.ELU(),
            nn.Linear(128, 64),
            nn.ELU(),
            nn.Linear(64, output_nbr),
        )

    def forward(self, x):
        x1 = self.logmel(x)
        x_env = self.env(x)
        x_env = x_env[:, :, :-1]
        x_env = self.cnn_env(x_env)

        a = self.cnn2d(x1)
        b = self.cnn1d(x_env)

        c = torch.cat((a, x_env.squeeze(3)), dim=1)
        x_flat = c.view(c.size(0), -1)
        z = self.fc(x_flat)
        return z

class v1_mi2(nn.Module):
    def __init__(self, output_nbr, sr):
        super(v1_mi2, self).__init__()

        self.logmel = LogMelSpectrogramLayer(sample_rate=sr, n_mels=256)

        self.fc = nn.Sequential(
            nn.Linear(160 * 2, 80),
            nn.ReLU(),
            nn.Linear(80, 40),
            nn.ReLU(),
            nn.Linear(40, output_nbr)
        )

        self.cnn1 = self._create_cnn_block()
        self.cnn2 = self._create_cnn_block()

    def _create_cnn_block(self):
        return nn.Sequential(
            custom2DCNN(1, 40, (2, 3), "same"),
            custom2DCNN(40, 40, (2, 3), "same"),
            nn.MaxPool2d((2, 1)),
            nn.Dropout2d(0.25),
            custom2DCNN(40, 80, (2, 3), "same"),
            custom2DCNN(80, 80, (2, 3), "same"),
            nn.MaxPool2d((2, 3)),
            nn.Dropout2d(0.25),
            custom2DCNN(80, 160, 2, "same"),
            nn.MaxPool2d((2, 1)),
            nn.Dropout2d(0.25),
            custom2DCNN(160, 160, 2, "same"),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.25),
            custom2DCNN(160, 160, 2, "same"),
            nn.MaxPool2d((2, 1)),
            nn.Dropout2d(0.25),
            custom2DCNN(160, 160, 2, "same"),
            nn.MaxPool2d((4, 2)),
            nn.Dropout2d(0.25),
        )

    def forward(self, x):
        x1, x2 = torch.split(self.logmel(x), 128, dim=2)

        x1 = self.cnn1(x1)
        x2 = self.cnn2(x2)

        x = torch.cat((x1, x2), dim=1)

        x_flat = x.view(x.size(0), -1)
        z = self.fc(x_flat)
        return z

class v1_mi4(nn.Module):
    def __init__(self, output_nbr, sr):
        super(v1_mi4, self).__init__()

        self.logmel = LogMelSpectrogramLayer(sample_rate=sr, n_mels=512)
        
        self.cnn1 = self._create_cnn_block()
        self.cnn2 = self._create_cnn_block()
        self.cnn3 = self._create_cnn_block()
        self.cnn4 = self._create_cnn_block()

        self.fc = nn.Sequential(
            nn.Linear(160 * 4, 240),
            nn.ReLU(),
            nn.Linear(240, 40),
            nn.ReLU(),
            nn.Linear(40, output_nbr)
        )

    def _create_cnn_block(self):
        return nn.Sequential(
            custom2DCNN(1, 40, (2, 3), "same"),
            custom2DCNN(40, 40, (2, 3), "same"),
            nn.MaxPool2d((2, 1)),
            nn.Dropout2d(0.25),
            custom2DCNN(40, 80, (2, 3), "same"),
            custom2DCNN(80, 80, (2, 3), "same"),
            nn.MaxPool2d((2, 3)),
            nn.Dropout2d(0.25),
            custom2DCNN(80, 160, 2, "same"),
            nn.MaxPool2d((2, 1)),
            nn.Dropout2d(0.25),
            custom2DCNN(160, 160, 2, "same"),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.25),
            custom2DCNN(160, 160, 2, "same"),
            nn.MaxPool2d((2, 1)),
            nn.Dropout2d(0.25),
            custom2DCNN(160, 160, 2, "same"),
            nn.MaxPool2d((4, 2)),
            nn.Dropout2d(0.25),
        )

    def forward(self, x):
        x1, x2, x3, x4 = torch.split(self.logmel(x), 128, dim=2)

        x1 = self.cnn1(x1)
        x2 = self.cnn2(x2)
        x3 = self.cnn3(x3)
        x4 = self.cnn4(x4)

        x = torch.cat((x1, x2, x3, x4), dim=1)

        x_flat = x.view(x.size(0), -1)
        z = self.fc(x_flat)
        return z
    
class v1_mi5_env2(nn.Module):
    def __init__(self, output_nbr, sr):
        super(v1_mi5_env2, self).__init__()

        self.logmel = LogMelSpectrogramLayer(sample_rate=sr, n_mels=640)
        self.env = EnvelopeFollowingLayerTorchScript(n_fft=2048, hop_length=512, smoothing_factor=4)
        
        self.cnn1 = self._create_cnn_block()
        self.cnn2 = self._create_cnn_block()
        self.cnn3 = self._create_cnn_block()
        self.cnn4 = self._create_cnn_block()
        self.cnn5 = self._create_cnn_block()
        self.cnn_env = self._create_cnn_env_block()

        self.fc = nn.Sequential(
            nn.Linear(160 * 6, 240),
            nn.ReLU(),
            nn.Linear(240, 60),
            nn.ReLU(),
            nn.Linear(60, output_nbr)
        )

    def _create_cnn_env_block(self):
        return nn.Sequential(
            custom1DCNN(1, 40, 7, "same", 4),
            nn.AvgPool1d(16),
            custom1DCNN(40, 40, 5, "same", 3),
            nn.AvgPool1d(8),
            custom1DCNN(40, 80, 3, "same", 2),
            nn.AvgPool1d(8),
            custom1DCNN(80, 160, 2, "same", 1),
            nn.AvgPool1d(7),
            nn.Dropout1d(0.1),
        )

    def _create_cnn_block(self):
        return nn.Sequential(
            custom2DCNN(1, 40, (2, 3), "same"),
            custom2DCNN(40, 40, (2, 3), "same"),
            nn.MaxPool2d((2, 1)),
            nn.Dropout2d(0.25),
            custom2DCNN(40, 80, (2, 3), "same"),
            custom2DCNN(80, 80, (2, 3), "same"),
            nn.MaxPool2d((2, 3)),
            nn.Dropout2d(0.25),
            custom2DCNN(80, 160, 2, "same"),
            nn.MaxPool2d((2, 1)),
            nn.Dropout2d(0.25),
            custom2DCNN(160, 160, 2, "same"),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.25),
            custom2DCNN(160, 160, 2, "same"),
            nn.MaxPool2d((2, 1)),
            nn.Dropout2d(0.25),
            custom2DCNN(160, 160, 2, "same"),
            nn.MaxPool2d((4, 2)),
            nn.Dropout2d(0.25),
        )

    def forward(self, x):
        x_env = self.env(x)
        x_env = x_env[:, :, :-1]
        x_env = self.cnn_env(x_env)

        x1, x2, x3, x4, x5 = torch.split(self.logmel(x), 128, dim=2)

        x1 = self.cnn1(x1)
        x2 = self.cnn2(x2)
        x3 = self.cnn3(x3)
        x4 = self.cnn4(x4)
        x5 = self.cnn5(x5)

        x = torch.cat((x1, x2, x3, x4, x5, x_env.unsqueeze(3)), dim=1)

        x_flat = x.view(x.size(0), -1)
        z = self.fc(x_flat)
        return z
    
class v1_mi6_env2(nn.Module):
    def __init__(self, output_nbr, args):
        super(v1_mi6_env2, self).__init__()

        self.sr = args.sr

        self.logmel = LogMelSpectrogramLayer(sample_rate=self.sr, n_mels=420)
        self.env = EnvelopeFollowingLayerTorchScript(n_fft=2048, hop_length=512, smoothing_factor=4)
        
        self.cnn1 = self._create_cnn_block()
        self.cnn2 = self._create_cnn_block()
        self.cnn3 = self._create_cnn_block()
        self.cnn4 = self._create_cnn_block()
        self.cnn5 = self._create_cnn_block()
        self.cnn6 = self._create_cnn_block()

        self.cnn_env = self._create_cnn_env_block()

        self.fc = nn.Sequential(
                nn.Linear(160 * 7, 320),
                nn.BatchNorm1d(320),
                nn.ReLU(),
                nn.Dropout1d(0.25),
                nn.Linear(320, 80),
                nn.BatchNorm1d(80),
                nn.ReLU(),
                nn.Dropout1d(0.25),
                nn.Linear(80, output_nbr)
            )

    def _create_cnn_env_block(self):
        return nn.Sequential(
            custom1DCNN(1, 40, 7, "same", 4),
            nn.AvgPool1d(16),
            nn.Dropout1d(0.25),
            custom1DCNN(40, 40, 5, "same", 3),
            nn.AvgPool1d(8),
            nn.Dropout1d(0.25),
            custom1DCNN(40, 80, 3, "same", 2),
            nn.AvgPool1d(8),
            nn.Dropout1d(0.25),
            custom1DCNN(80, 160, 2, "same", 1),
            nn.AvgPool1d(7),
            nn.Dropout1d(0.25),
        )

    def _create_cnn_block(self):
        return nn.Sequential(
            custom2DCNN(1, 40, (2, 3), "same"),
            custom2DCNN(40, 40, (2, 3), "same"),
            nn.MaxPool2d((2, 1)), # 35
            nn.Dropout2d(0.25),
            custom2DCNN(40, 80, (2, 3), "same"),
            custom2DCNN(80, 80, (2, 3), "same"),
            nn.MaxPool2d((2, 3)), # 17
            nn.Dropout2d(0.25),
            custom2DCNN(80, 160, 2, "same"),
            nn.MaxPool2d((2, 1)), # 8
            nn.Dropout2d(0.25),
            custom2DCNN(160, 160, 2, "same"),
            nn.MaxPool2d(2), # 4
            nn.Dropout2d(0.25),
            custom2DCNN(160, 160, 2, "same"),
            nn.MaxPool2d((2, 1)), #2
            nn.Dropout2d(0.25),
            custom2DCNN(160, 160, 2, "same"),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.25),
        )

    def forward(self, x):
        x_env = self.env(x)
        x_env = x_env[:, :, :-1]
        x_env = self.cnn_env(x_env)

        x1, x2, x3, x4, x5, x6 = torch.split(self.logmel(x), 70, dim=2)

        x1 = self.cnn1(x1) 
        x2 = self.cnn2(x2)
        x3 = self.cnn3(x3)
        x4 = self.cnn4(x4)
        x5 = self.cnn5(x5)
        x6 = self.cnn6(x6)

        x = torch.cat((x1, x2, x3, x4, x5, x6, x_env.unsqueeze(3)), dim=1)

        x_flat = x.view(x.size(0), -1)
        z = self.fc(x_flat)
        return z

class v1_mi6_env2_mod(nn.Module):
    def __init__(self, output_nbr, args):
        super(v1_mi6_env2_mod, self).__init__()

        self.sr = args.sr

        self.logmel = LogMelSpectrogramLayer(sample_rate=self.sr, n_mels=420)
        self.env = EnvelopeFollowingLayerTorchScript(n_fft=2048, hop_length=512, smoothing_factor=4)
        
        self.cnn1 = self._create_cnn_block()
        self.cnn2 = self._create_cnn_block()
        self.cnn3 = self._create_cnn_block()
        self.cnn4 = self._create_cnn_block()
        self.cnn5 = self._create_cnn_block()
        self.cnn6 = self._create_cnn_block()

        self.cnn_env = self._create_cnn_env_block()

        self.fc = nn.Sequential(
                nn.Linear(160 * 7, 320),
                nn.BatchNorm1d(320),
                nn.ReLU(),
                nn.Dropout1d(0.2),
                nn.Linear(320, 80),
                nn.BatchNorm1d(80),
                nn.ReLU(),
                nn.Dropout1d(0.2),
                nn.Linear(80, output_nbr)
            )

    def _create_cnn_env_block(self):
        return nn.Sequential(
            custom1DCNN(1, 40, 7, "same", 4),
            nn.AvgPool1d(16),
            nn.Dropout1d(0.25),
            custom1DCNN(40, 40, 5, "same", 3),
            nn.AvgPool1d(8),
            nn.Dropout1d(0.25),
            custom1DCNN(40, 80, 3, "same", 2),
            nn.AvgPool1d(8),
            nn.Dropout1d(0.25),
            custom1DCNN(80, 160, 2, "same", 1),
            nn.AvgPool1d(7),
            nn.Dropout1d(0.25),
        )

    def _create_cnn_block(self):
        return nn.Sequential(
            custom2DCNN(1, 40, (2, 7), "same"),
            custom2DCNN(40, 40, (2, 7), "same"),
            nn.MaxPool2d((2, 1)), # 35, 15
            nn.Dropout2d(0.25),
            custom2DCNN(40, 80, (2, 5), "same"),
            custom2DCNN(80, 80, (2, 5), "same"),
            nn.MaxPool2d((2, 3)), # 17, 5
            nn.Dropout2d(0.25),
            custom2DCNN(80, 160, (2, 3), "same"),
            nn.MaxPool2d((2, 1)), # 8, 5
            nn.Dropout2d(0.25),
            custom2DCNN(160, 160, (2, 3), "same"),
            nn.MaxPool2d(2), # 4, 2
            nn.Dropout2d(0.25),
            custom2DCNN(160, 160, (1, 2), "same"),
            nn.MaxPool2d((2, 1)), #2, 2
            nn.Dropout2d(0.25),
            custom2DCNN(160, 160, (1, 2), "same"),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.25),
        )

    # def _create_cnn_block(self):
    #     return nn.Sequential(
    #         custom2DCNN(1, 40, (2, 5), "same"),
    #         custom2DCNN(40, 40, (2, 5), "same"),
    #         nn.MaxPool2d((2, 1)), # 35
    #         nn.Dropout2d(0.25),
    #         custom2DCNN(40, 80, (2, 3), "same"),
    #         custom2DCNN(80, 80, (2, 3), "same"),
    #         nn.MaxPool2d((2, 3)), # 17
    #         nn.Dropout2d(0.25),
    #         custom2DCNN(80, 160, 2, "same"),
    #         nn.MaxPool2d((2, 1)), # 8
    #         nn.Dropout2d(0.25),
    #         custom2DCNN(160, 160, 2, "same"),
    #         nn.MaxPool2d(2), # 4
    #         nn.Dropout2d(0.25),
    #         custom2DCNN(160, 160, 2, "same"),
    #         nn.MaxPool2d((2, 1)), #2
    #         nn.Dropout2d(0.25),
    #         custom2DCNN(160, 160, 2, "same"),
    #         nn.MaxPool2d(2),
    #         nn.Dropout2d(0.25),
    #     )

    def forward(self, x):
        x_env = self.env(x)
        x_env = x_env[:, :, :-1]
        x_env = self.cnn_env(x_env)

        x1, x2, x3, x4, x5, x6 = torch.split(self.logmel(x), 70, dim=2)

        x1 = self.cnn1(x1) 
        x2 = self.cnn2(x2)
        x3 = self.cnn3(x3)
        x4 = self.cnn4(x4)
        x5 = self.cnn5(x5)
        x6 = self.cnn6(x6)

        x = torch.cat((x1, x2, x3, x4, x5, x6, x_env.unsqueeze(3)), dim=1)

        x_flat = x.view(x.size(0), -1)
        z = self.fc(x_flat)
        return z
    
class v1_mi6_env2_stack(nn.Module):
    def __init__(self, output_nbr, args):
        super(v1_mi6_env2_stack, self).__init__()

        self.sr = args.sr

        self.logmel1 = LogMelSpectrogramLayer(sample_rate=self.sr, n_mels=420, n_fft=2048, hop_length=512)
        self.logmel2 = LogMelSpectrogramLayer(sample_rate=self.sr, n_mels=420, n_fft=4096, hop_length=512)
        self.logmel3 = LogMelSpectrogramLayer(sample_rate=self.sr, n_mels=420, n_fft=1024, hop_length=512)
        self.env = EnvelopeFollowingLayerTorchScript(n_fft=2048, hop_length=512, smoothing_factor=4)
        
        self.cnn1 = self._create_cnn_block()
        self.cnn2 = self._create_cnn_block()
        self.cnn3 = self._create_cnn_block()
        self.cnn4 = self._create_cnn_block()
        self.cnn5 = self._create_cnn_block()
        self.cnn6 = self._create_cnn_block()

        self.cnn_env = self._create_cnn_env_block()

        self.fc = nn.Sequential(
                nn.Linear(160 * 7, 320),
                nn.BatchNorm1d(320),
                nn.ReLU(),
                nn.Dropout1d(0.25),
                nn.Linear(320, 80),
                nn.BatchNorm1d(80),
                nn.ReLU(),
                nn.Dropout1d(0.25),
                nn.Linear(80, output_nbr)
            )

    def _create_cnn_env_block(self):
        return nn.Sequential(
            custom1DCNN(1, 40, 7, "same", 4),
            nn.AvgPool1d(16),
            nn.Dropout1d(0.25),
            custom1DCNN(40, 40, 5, "same", 3),
            nn.AvgPool1d(8),
            nn.Dropout1d(0.25),
            custom1DCNN(40, 80, 3, "same", 2),
            nn.AvgPool1d(8),
            nn.Dropout1d(0.25),
            custom1DCNN(80, 160, 2, "same", 1),
            nn.AvgPool1d(7),
            nn.Dropout1d(0.25),
        )

    def _create_cnn_block(self):
        return nn.Sequential(
            custom2DCNN(3, 40, (2, 3), "same"),
            custom2DCNN(40, 40, (2, 3), "same"),
            nn.MaxPool2d((2, 1)), # 35
            nn.Dropout2d(0.25),
            custom2DCNN(40, 80, (2, 3), "same"),
            custom2DCNN(80, 80, (2, 3), "same"),
            nn.MaxPool2d((2, 3)), # 17
            nn.Dropout2d(0.25),
            custom2DCNN(80, 160, 2, "same"),
            nn.MaxPool2d((2, 1)), # 8
            nn.Dropout2d(0.25),
            custom2DCNN(160, 160, 2, "same"),
            nn.MaxPool2d(2), # 4
            nn.Dropout2d(0.25),
            custom2DCNN(160, 160, 2, "same"),
            nn.MaxPool2d((2, 1)), #2
            nn.Dropout2d(0.25),
            custom2DCNN(160, 160, 2, "same"),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.25),
        )

    def forward(self, x):
        x_env = self.env(x)
        x_env = x_env[:, :, :-1]
        x_env = self.cnn_env(x_env)

        x1_1, x1_2, x1_3, x1_4, x1_5, x1_6 = torch.split(self.logmel1(x), 70, dim=2)
        x2_1, x2_2, x2_3, x2_4, x2_5, x2_6 = torch.split(self.logmel2(x), 70, dim=2)
        x3_1, x3_2, x3_3, x3_4, x3_5, x3_6 = torch.split(self.logmel3(x), 70, dim=2)

        rx1_s = torch.cat((x1_1, x2_1, x3_1), dim=1)
        rx2_s = torch.cat((x1_2, x2_2, x3_2), dim=1)
        rx3_s = torch.cat((x1_3, x2_3, x3_3), dim=1)
        rx4_s = torch.cat((x1_4, x2_4, x3_4), dim=1)
        rx5_s = torch.cat((x1_5, x2_5, x3_5), dim=1)
        rx6_s = torch.cat((x1_6, x2_6, x3_6), dim=1)

        x1 = self.cnn1(rx1_s) 
        x2 = self.cnn2(rx2_s)
        x3 = self.cnn3(rx3_s)
        x4 = self.cnn4(rx4_s)
        x5 = self.cnn5(rx5_s)
        x6 = self.cnn6(rx6_s)

        x = torch.cat((x1, x2, x3, x4, x5, x6, x_env.unsqueeze(3)), dim=1)

        x_flat = x.view(x.size(0), -1)
        z = self.fc(x_flat)
        return z
    
class v1_mi6_env2_mod_stacks(nn.Module):
    def __init__(self, output_nbr, args):
        super(v1_mi6_env2_mod_stacks, self).__init__()

        self.sr = args.sr

        self.logmel1 = LogMelSpectrogramLayer(sample_rate=self.sr, n_mels=420, n_fft=2048, hop_length=512)
        self.logmel2 = LogMelSpectrogramLayer(sample_rate=self.sr, n_mels=420, n_fft=2048, hop_length=256)
        self.env = EnvelopeFollowingLayerTorchScript(n_fft=2048, hop_length=512, smoothing_factor=128)
        
        self.cnn1 = self._create_cnn_block()
        self.cnn2 = self._create_cnn_block()
        self.cnn3 = self._create_cnn_block()
        self.cnn4 = self._create_cnn_block()
        self.cnn5 = self._create_cnn_block()
        self.cnn6 = self._create_cnn_block()

        self.cnn_env = self._create_cnn_env_block()

        self.fc = nn.Sequential(
                nn.Linear(160 * 7, 320),
                nn.BatchNorm1d(320),
                nn.ReLU(),
                nn.Dropout1d(0.25),
                nn.Linear(320, 80),
                nn.BatchNorm1d(80),
                nn.ReLU(),
                nn.Dropout1d(0.25),
                nn.Linear(80, output_nbr)
            )
    def _create_cnn_env_block(self):
        return nn.Sequential(
            custom1DCNN(1, 40, 7, "same", 1),
            nn.AvgPool1d(16),
            nn.Dropout1d(0.25),
            custom1DCNN(40, 40, 5, "same", 2),
            nn.AvgPool1d(8),
            nn.Dropout1d(0.25),
            custom1DCNN(40, 80, 3, "same", 3),
            nn.AvgPool1d(8),
            nn.Dropout1d(0.25),
            custom1DCNN(80, 160, 2, "same", 4),
            nn.AvgPool1d(7),
            nn.Dropout1d(0.25),
        )

    def _create_cnn_block(self):
        return nn.Sequential(
            custom2DCNN(3, 40, (3, 7), "same"),
            custom2DCNN(40, 40, (3, 7), "same"),
            nn.MaxPool2d((2, 1)), # 35, 15
            nn.Dropout2d(0.25),
            custom2DCNN(40, 80, (2, 5), "same"),
            custom2DCNN(80, 80, (2, 5), "same"),
            nn.MaxPool2d((2, 3)), # 17, 5
            nn.Dropout2d(0.25),
            custom2DCNN(80, 160, (2, 3), "same"),
            nn.MaxPool2d((2, 1)), # 8, 5
            nn.Dropout2d(0.25),
            custom2DCNN(160, 160, (1, 3), "same"),
            nn.MaxPool2d(2), # 4, 2
            nn.Dropout2d(0.25),
            custom2DCNN(160, 160, (1, 2), "same"),
            nn.MaxPool2d((2, 1)), #2, 2
            nn.Dropout2d(0.25),
            custom2DCNN(160, 160, (1, 2), "same"),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.25)
        )

    def forward(self, x):
        x_env = self.env(x)
        x_env = x_env[:, :, :-1]
        x_env = self.cnn_env(x_env)

        x1_1, x1_2, x1_3, x1_4, x1_5, x1_6 = torch.split(self.logmel1(x), 70, dim=2)
        x2_1, x2_2, x2_3, x2_4, x2_5, x2_6 = torch.split(self.logmel2(x)[:,:,:, :15], 70, dim=2)
        x3_1, x3_2, x3_3, x3_4, x3_5, x3_6 = torch.split(self.logmel2(x)[:,:,:, 14:], 70, dim=2)

        rx1_s = torch.cat((x1_1, x2_1, x3_1), dim=1)
        rx2_s = torch.cat((x1_2, x2_2, x3_2), dim=1)
        rx3_s = torch.cat((x1_3, x2_3, x3_3), dim=1)
        rx4_s = torch.cat((x1_4, x2_4, x3_4), dim=1)
        rx5_s = torch.cat((x1_5, x2_5, x3_5), dim=1)
        rx6_s = torch.cat((x1_6, x2_6, x3_6), dim=1)

        x1 = self.cnn1(rx1_s) 
        x2 = self.cnn2(rx2_s)
        x3 = self.cnn3(rx3_s)
        x4 = self.cnn4(rx4_s)
        x5 = self.cnn5(rx5_s)
        x6 = self.cnn6(rx6_s)

        x = torch.cat((x1, x2, x3, x4, x5, x6, x_env.unsqueeze(3)), dim=1)

        x_flat = x.view(x.size(0), -1)
        z = self.fc(x_flat)
        return z
    
class v1_mi6_env2_stacks7(nn.Module):
    def __init__(self, output_nbr, args):
        super(v1_mi6_env2_stacks7, self).__init__()

        self.sr = args.sr

        self.logmel1 = LogMelSpectrogramLayer(sample_rate=self.sr, n_mels=420, n_fft=2048, hop_length=512)
        self.logmel2 = LogMelSpectrogramLayer(sample_rate=self.sr, n_mels=420, n_fft=2048, hop_length=256)
        self.logmel3 = LogMelSpectrogramLayer(sample_rate=self.sr, n_mels=420, n_fft=2048, hop_length=128)
        self.env = EnvelopeFollowingLayerTorchScript(n_fft=2048, hop_length=512, smoothing_factor=4)
        
        self.cnn1 = self._create_cnn_block()
        self.cnn2 = self._create_cnn_block()
        self.cnn3 = self._create_cnn_block()
        self.cnn4 = self._create_cnn_block()
        self.cnn5 = self._create_cnn_block()
        self.cnn6 = self._create_cnn_block()

        self.cnn_env = self._create_cnn_env_block()

        self.fc = nn.Sequential(
                nn.Linear(160 * 7, 320),
                nn.BatchNorm1d(320),
                nn.ReLU(),
                nn.Dropout1d(0.25),
                nn.Linear(320, 80),
                nn.BatchNorm1d(80),
                nn.ReLU(),
                nn.Dropout1d(0.25),
                nn.Linear(80, output_nbr)
            )

    def _create_cnn_env_block(self):
        return nn.Sequential(
            custom1DCNN(1, 40, 7, "same", 4),
            nn.AvgPool1d(16),
            nn.Dropout1d(0.25),
            custom1DCNN(40, 40, 5, "same", 3),
            nn.AvgPool1d(8),
            nn.Dropout1d(0.25),
            custom1DCNN(40, 80, 3, "same", 2),
            nn.AvgPool1d(8),
            nn.Dropout1d(0.25),
            custom1DCNN(80, 160, 2, "same", 1),
            nn.AvgPool1d(7),
            nn.Dropout1d(0.25),
        )

    def _create_cnn_block(self):
        return nn.Sequential(
            custom2DCNN(7, 40, (2, 3), "same"),
            custom2DCNN(40, 40, (2, 3), "same"),
            nn.MaxPool2d((2, 1)), # 35
            nn.Dropout2d(0.25),
            custom2DCNN(40, 80, (2, 3), "same"),
            custom2DCNN(80, 80, (2, 3), "same"),
            nn.MaxPool2d((2, 3)), # 17
            nn.Dropout2d(0.25),
            custom2DCNN(80, 160, 2, "same"),
            nn.MaxPool2d((2, 1)), # 8
            nn.Dropout2d(0.25),
            custom2DCNN(160, 160, 2, "same"),
            nn.MaxPool2d(2), # 4
            nn.Dropout2d(0.25),
            custom2DCNN(160, 160, 2, "same"),
            nn.MaxPool2d((2, 1)), #2
            nn.Dropout2d(0.25),
            custom2DCNN(160, 160, 2, "same"),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.25),
        )

    def forward(self, x):
        x_env = self.env(x)
        x_env = x_env[:, :, :-1]
        x_env = self.cnn_env(x_env)

        x1_1, x1_2, x1_3, x1_4, x1_5, x1_6 = torch.split(self.logmel1(x), 70, dim=2)
        x2_1, x2_2, x2_3, x2_4, x2_5, x2_6 = torch.split(self.logmel2(x)[:,:,:, :15], 70, dim=2)
        x3_1, x3_2, x3_3, x3_4, x3_5, x3_6 = torch.split(self.logmel2(x)[:,:,:, 14:], 70, dim=2)

        x4_1, x4_2, x4_3, x4_4, x4_5, x4_6 = torch.split(self.logmel3(x)[:,:,:, :15], 70, dim=2)
        x5_1, x5_2, x5_3, x5_4, x5_5, x5_6 = torch.split(self.logmel3(x)[:,:,:, 13:28], 70, dim=2)
        x6_1, x6_2, x6_3, x6_4, x6_5, x6_6 = torch.split(self.logmel3(x)[:,:,:, 27:42], 70, dim=2)
        x7_1, x7_2, x7_3, x7_4, x7_5, x7_6 = torch.split(self.logmel3(x)[:,:,:, 42:], 70, dim=2)
        
        rx1_s = torch.cat((x1_1, x2_1, x3_1, x4_1, x5_1, x6_1, x7_1), dim=1)
        rx2_s = torch.cat((x1_2, x2_2, x3_2, x4_2, x5_2, x6_2, x7_2), dim=1)
        rx3_s = torch.cat((x1_3, x2_3, x3_3, x4_3, x5_3, x6_3, x7_3), dim=1)
        rx4_s = torch.cat((x1_4, x2_4, x3_4, x4_4, x5_4, x6_4, x7_4), dim=1)
        rx5_s = torch.cat((x1_5, x2_5, x3_5, x4_5, x5_5, x6_5, x7_5), dim=1)
        rx6_s = torch.cat((x1_6, x2_6, x3_6, x4_6, x5_6, x6_6, x7_6), dim=1)

        x1 = self.cnn1(rx1_s) 
        x2 = self.cnn2(rx2_s)
        x3 = self.cnn3(rx3_s)
        x4 = self.cnn4(rx4_s)
        x5 = self.cnn5(rx5_s)
        x6 = self.cnn6(rx6_s)

        x = torch.cat((x1, x2, x3, x4, x5, x6, x_env.unsqueeze(3)), dim=1)

        x_flat = x.view(x.size(0), -1)
        z = self.fc(x_flat)
        return z
    
# old
class v1_mi6_env2_lstm(nn.Module):
    def __init__(self, output_nbr, args):
        super(v1_mi6_env2_lstm, self).__init__()

        self.sr = args.sr

        self.logmel = LogMelSpectrogramLayer(sample_rate=self.sr, n_mels=420, hop_length=512)
        self.env = EnvelopeFollowingLayerTorchScript(n_fft=2048, hop_length=512, smoothing_factor=1024)
        
        self.cnn1 = self._create_cnn_block()
        self.cnn2 = self._create_cnn_block()
        self.cnn3 = self._create_cnn_block()
        self.cnn4 = self._create_cnn_block()
        self.cnn5 = self._create_cnn_block()
        self.cnn6 = self._create_cnn_block()

        self.cnn_env = self._create_cnn_env_block()
        self.lstm_env = self._create_lstm_env_block()

        self.fc = nn.Sequential(
            nn.Linear(112 * 7, 260),
            nn.ReLU(),
            nn.Linear(260, 140), 
            nn.ReLU(),
            nn.Linear(140, output_nbr)
        )

    def _create_cnn_env_block(self):
        return nn.Sequential(
            custom1DCNN(1, 64, 7, "same", 4),
            nn.AvgPool1d(8),
            custom1DCNN(64, 128, 5, "same", 3),
            nn.AvgPool1d(8),
            custom1DCNN(128, 256, 2, "same", 1),
        )
    
    def _create_lstm_env_block(self):
        return nn.LSTM(input_size=256, hidden_size=112, batch_first=True)
    

    def _create_cnn_block(self):
        return nn.Sequential(
            custom2DCNN(1, 28, (2, 3), "same"),
            custom2DCNN(28, 28, (2, 3), "same"),
            nn.MaxPool2d((2, 1)), # 35
            nn.Dropout2d(0.25),
            custom2DCNN(28, 56, (2, 3), "same"),
            custom2DCNN(56, 56, (2, 3), "same"),
            nn.MaxPool2d((2, 3)), # 17
            nn.Dropout2d(0.25),
            custom2DCNN(56, 112, 2, "same"),
            nn.MaxPool2d((2, 1)), # 8
            nn.Dropout2d(0.25),
            custom2DCNN(112, 112, 2, "same"),
            nn.MaxPool2d(2), # 4
            nn.Dropout2d(0.25),
            custom2DCNN(112, 112, 2, "same"),
            nn.MaxPool2d((2, 1)), #2
            nn.Dropout2d(0.25),
            custom2DCNN(112, 112, 2, "same"),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.25),
        )

    def forward(self, x):
        x_env = self.env(x)
        x_env = x_env[:, :, :-1]
        # print(x_env)
        x_env = self.cnn_env(x_env)

        x_env = x_env.permute(0, 2, 1)
        # print(x_env.shape)
        lstm_out = self.lstm_env(x_env)
        lstm_out_last = lstm_out[0][:, -1, :].unsqueeze(2).unsqueeze(2)
        # print(lstm_out_last.shape)

        x1, x2, x3, x4, x5, x6 = torch.split(self.logmel(x), 70, dim=2)

        x1 = self.cnn1(x1) 
        x2 = self.cnn2(x2)
        x3 = self.cnn3(x3)
        x4 = self.cnn4(x4)
        x5 = self.cnn5(x5)
        x6 = self.cnn6(x6)

        x = torch.cat((x1, x2, x3, x4, x5, x6, lstm_out_last), dim=1)

        x_flat = x.view(x.size(0), -1)
        z = self.fc(x_flat)
        return z
    

class v1_mi6_env2_lstm_new(nn.Module):
    def __init__(self, output_nbr, args):
        super(v1_mi6_env2_lstm_new, self).__init__()

        self.sr = args.sr

        self.logmel = LogMelSpectrogramLayer(sample_rate=self.sr, n_mels=420, hop_length=512)
        self.env = EnvelopeFollowingLayerTorchScript(n_fft=2048, hop_length=512, smoothing_factor=4)
        
        self.cnn1 = self._create_cnn_block()
        self.cnn2 = self._create_cnn_block()
        self.cnn3 = self._create_cnn_block()
        self.cnn4 = self._create_cnn_block()
        self.cnn5 = self._create_cnn_block()
        self.cnn6 = self._create_cnn_block()

        self.cnn_env = self._create_cnn_env_block()
        self.lstm_env = self._create_lstm_env_block()

        self.fc = nn.Sequential(
            nn.Linear(128 * 7, 256),
            nn.ReLU(),
            nn.Linear(256, 64), 
            nn.ReLU(),
            nn.Linear(64, output_nbr)
        )

    def _create_cnn_env_block(self):
        return nn.Sequential(
            custom1DCNN(1, 64, 7, "same", 4),
            nn.AvgPool1d(8),
            custom1DCNN(64, 128, 5, "same", 3),
            nn.AvgPool1d(8),
            custom1DCNN(128, 256, 2, "same", 1),
        )

    def _create_lstm_env_block(self):
        return nn.LSTM(input_size=256, hidden_size=112, batch_first=True)
    

    def _create_cnn_block(self):
        return nn.Sequential(
            custom2DCNN(1, 32, (2, 3), "same"),
            custom2DCNN(32, 32, (2, 3), "same"),
            nn.MaxPool2d((2, 1)), # 35
            nn.Dropout2d(0.25),
            custom2DCNN(32, 64, (2, 3), "same"),
            custom2DCNN(64, 64, (2, 3), "same"),
            nn.MaxPool2d((2, 3)), # 17
            nn.Dropout2d(0.25),
            custom2DCNN(64, 128, 2, "same"),
            nn.MaxPool2d((2, 1)), # 8
            nn.Dropout2d(0.25),
            custom2DCNN(128, 128, 2, "same"),
            nn.MaxPool2d(2), # 4
            nn.Dropout2d(0.25),
            custom2DCNN(128, 128, 2, "same"),
            nn.MaxPool2d((2, 1)), #2
            nn.Dropout2d(0.25),
            custom2DCNN(128, 128, 2, "same"),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.25),
        )

    def forward(self, x):
        x_env = self.env(x)
        x_env = x_env[:, :, :-1]
        x_env = self.cnn_env(x_env)

        x_env = x_env.permute(0, 2, 1)
        lstm_out = self.lstm_env(x_env)
        lstm_out_last = lstm_out[0][:, -1, :].unsqueeze(2).unsqueeze(2)

        x1, x2, x3, x4, x5, x6 = torch.split(self.logmel(x), 70, dim=2)

        x1 = self.cnn1(x1) 
        x2 = self.cnn2(x2)
        x3 = self.cnn3(x3)
        x4 = self.cnn4(x4)
        x5 = self.cnn5(x5)
        x6 = self.cnn6(x6)

        x = torch.cat((x1, x2, x3, x4, x5, x6, lstm_out_last), dim=1)

        x_flat = x.view(x.size(0), -1)
        z = self.fc(x_flat)
        return z


class v1_mi6(nn.Module):
    def __init__(self, output_nbr, sr):
        super(v1_mi6, self).__init__()

        self.logmel = LogMelSpectrogramLayer(sample_rate=sr, n_mels=420)
        
        self.cnn1 = self._create_cnn_block()
        self.cnn2 = self._create_cnn_block()
        self.cnn3 = self._create_cnn_block()
        self.cnn4 = self._create_cnn_block()
        self.cnn5 = self._create_cnn_block()
        self.cnn6 = self._create_cnn_block()

        self.fc = nn.Sequential(
            nn.Linear(160 * 6, 320),
            nn.ReLU(),
            nn.Linear(320, 80),
            nn.ReLU(),
            nn.Linear(80, output_nbr)
        )

    def _create_cnn_block(self):
        return nn.Sequential(
            custom2DCNN(1, 40, (2, 3), "same"),
            custom2DCNN(40, 40, (2, 3), "same"),
            nn.MaxPool2d((2, 1)), # 35
            nn.Dropout2d(0.25),
            custom2DCNN(40, 80, (2, 3), "same"),
            custom2DCNN(80, 80, (2, 3), "same"),
            nn.MaxPool2d((2, 3)), # 17
            nn.Dropout2d(0.25),
            custom2DCNN(80, 160, 2, "same"),
            nn.MaxPool2d((2, 1)), # 8
            nn.Dropout2d(0.25),
            custom2DCNN(160, 160, 2, "same"),
            nn.MaxPool2d(2), # 4
            nn.Dropout2d(0.25),
            custom2DCNN(160, 160, 2, "same"),
            nn.MaxPool2d((2, 1)), #2
            nn.Dropout2d(0.25),
            custom2DCNN(160, 160, 2, "same"),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.25),
        )

    def forward(self, x):
        x1, x2, x3, x4, x5, x6 = torch.split(self.logmel(x), 70, dim=2)

        x1 = self.cnn1(x1)
        x2 = self.cnn2(x2)
        x3 = self.cnn3(x3)
        x4 = self.cnn4(x4)
        x5 = self.cnn5(x5)
        x6 = self.cnn6(x6)

        x = torch.cat((x1, x2, x3, x4, x5, x6), dim=1)

        x_flat = x.view(x.size(0), -1)
        z = self.fc(x_flat)
        return z
    
# OLD: 20241009_001345
# class v1_mi6_hpss(nn.Module):
#     def __init__(self, output_nbr, args):
#         super(v1_mi6_hpss, self).__init__()

#         self.sr = args.sr
#         self.device = args.device

#         self.logmel = LogMelSpectrogramLayer(sample_rate=self.sr, n_mels=128, n_fft=2048, hop_length=512)
#         self.env = EnvelopeFollowingLayerTorchScript(n_fft=2048, hop_length=512, smoothing_factor=4)
#         self.hpss = HPSS(n_fft=2048, hop_length=512, kernel_size=31, device=self.device)
        
#         self.cnn1 = self._create_cnn_block()
#         self.cnn2 = self._create_cnn_block()
#         # self.cnn3 = self._create_cnn_block()
#         # self.cnn4 = self._create_cnn_block()
#         # self.cnn5 = self._create_cnn_block()
#         # self.cnn6 = self._create_cnn_block()

#         self.cnn_env = self._create_cnn_env_block()

#         self.fc = nn.Sequential(
#             nn.Linear(160 * 3, 160),
#             nn.ReLU(),
#             nn.Linear(160, 40),
#             nn.ReLU(),
#             nn.Linear(40, output_nbr)
#         )

#     def _create_cnn_env_block(self):
#         return nn.Sequential(
#             custom1DCNN(1, 40, 7, "same", 4),
#             nn.AvgPool1d(32), #avant: 16
#             # custom1DCNN(40, 40, 5, "same", 3),
#             # nn.AvgPool1d(8), 
#             custom1DCNN(40, 80, 5, "same", 3),
#             nn.AvgPool1d(16), #avant: 8
#             custom1DCNN(80, 160, 2, "same", 1),
#             nn.AvgPool1d(14),  #avant: 76
#             nn.Dropout1d(0.1),
#         )

#     def _create_cnn_block(self):
#         return nn.Sequential(
#             custom2DCNN(1, 40, (2, 3), "same"),
#             custom2DCNN(40, 40, (2, 3), "same"),
#             nn.MaxPool2d((2, 1)), # 64, 15
#             nn.Dropout2d(0.25),
#             custom2DCNN(40, 80, (2, 3), "same"),
#             custom2DCNN(80, 80, (2, 3), "same"),
#             nn.MaxPool2d((2, 3)), # 32, 5
#             nn.Dropout2d(0.25),
#             custom2DCNN(80, 160, 2, "same"),
#             nn.MaxPool2d((2, 1)), # 16, 5
#             nn.Dropout2d(0.25),
#             custom2DCNN(160, 160, 2, "same"),
#             nn.MaxPool2d(2), # 8, 2
#             nn.Dropout2d(0.25),
#             custom2DCNN(160, 160, 2, "same"),
#             nn.MaxPool2d((2, 1)), #4, 2
#             nn.Dropout2d(0.25),
#             custom2DCNN(160, 160, 2, "same"),
#             nn.MaxPool2d((4, 2)),
#             nn.Dropout2d(0.25),
#         )

#     def forward(self, x):
#         x_harm, x_perc = self.hpss(x)
#         # x1, x2, x3 = torch.split(self.logmel(x_harm), 70, dim=2)
#         # x4, x5, x6 = torch.split(self.logmel(x_perc), 70, dim=2)
#         x1 = self.logmel(x_harm)
#         x2 = self.logmel(x_perc)

#         x_env = self.env(x)
#         x_env = x_env[:, :, :-1]
#         x_env = self.cnn_env(x_env)

#         x1 = self.cnn1(x1) 
#         x2 = self.cnn2(x2)
#         # x3 = self.cnn3(x3)
#         # x4 = self.cnn4(x4)
#         # x5 = self.cnn5(x5)
#         # x6 = self.cnn6(x6)

#         x = torch.cat((x1, x2, x_env.unsqueeze(3)), dim=1)

#         x_flat = x.view(x.size(0), -1)
#         z = self.fc(x_flat)
#         return z

class v1_mi6_hpss(nn.Module):
    def __init__(self, output_nbr, args):
        super(v1_mi6_hpss, self).__init__()

        self.sr = args.sr
        self.device = args.device

        self.logmel = LogMelSpectrogramLayer(sample_rate=self.sr, n_mels=128, n_fft=2048, hop_length=512)
        self.env = EnvelopeFollowingLayerTorchScript(n_fft=2048, hop_length=512, smoothing_factor=512)
        self.hpss = HPSS(n_fft=2048, hop_length=512, kernel_size=31, device=self.device)
        
        self.cnn1 = self._create_cnn_block()
        self.cnn2 = self._create_cnn_block()
        # self.cnn3 = self._create_cnn_block()
        # self.cnn4 = self._create_cnn_block()
        # self.cnn5 = self._create_cnn_block()
        # self.cnn6 = self._create_cnn_block()

        self.cnn_env = self._create_cnn_env_block()

        self.fc = nn.Sequential(
            nn.Linear(160 * 3, 160),
            nn.ReLU(),
            nn.Linear(160, 40),
            nn.ReLU(),
            nn.Linear(40, output_nbr)
        )

    def _create_cnn_env_block(self):
        return nn.Sequential(
            nn.AvgPool1d(16),
            custom1DCNN(1, 40, 7, "same", 4),
            nn.AvgPool1d(8),
            nn.Dropout1d(0.25),
            custom1DCNN(40, 80, 5, "same", 3),
            nn.AvgPool1d(8),
            nn.Dropout1d(0.25),
            custom1DCNN(80, 160, 3, "same", 2),
            nn.AvgPool1d(7),
            nn.Dropout1d(0.25),
            # custom1DCNN(80, 160, 2, "same", 1),
            # nn.AvgPool1d(4), 
            # nn.Dropout1d(0.25),
        )
    
    # def _create_cnn_env_block(self):
        # return nn.Sequential(
        #     custom1DCNN(1, 20, 7, "same", 4),
        #     nn.AvgPool1d(16),
        #     nn.Dropout1d(0.25),
        #     custom1DCNN(20, 40, 5, "same", 3),
        #     nn.AvgPool1d(8), 
        #     nn.Dropout1d(0.25),
        #     custom1DCNN(40, 80, 3, "same", 2),
        #     nn.AvgPool1d(8),
        #     nn.Dropout1d(0.25),
        #     custom1DCNN(80, 160, 2, "same", 1),
        #     nn.AvgPool1d(7), 
        #     nn.Dropout1d(0.25),
        # )

    # def _create_cnn_env_block(self):
    #     return nn.Sequential(
    #         custom1DCNN(1, 40, 7, "same", 4),
    #         nn.AvgPool1d(16),
    #         nn.Dropout1d(0.1),
    #         custom1DCNN(40, 40, 5, "same", 3),
    #         nn.AvgPool1d(8), 
    #         nn.Dropout1d(0.1),
    #         custom1DCNN(40, 80, 3, "same", 2),
    #         nn.AvgPool1d(8),
    #         nn.Dropout1d(0.1),
    #         custom1DCNN(80, 160, 2, "same", 1),
    #         nn.AvgPool1d(7), 
    #         nn.Dropout1d(0.1),
    #     )

    def _create_cnn_block(self):
        return nn.Sequential(
            custom2DCNN(1, 40, (2, 3), "same"),
            custom2DCNN(40, 40, (2, 3), "same"),
            nn.MaxPool2d((2, 1)), # 64, 15
            nn.Dropout2d(0.25),
            custom2DCNN(40, 80, (2, 3), "same"),
            custom2DCNN(80, 80, (2, 3), "same"),
            nn.MaxPool2d((2, 3)), # 32, 5
            nn.Dropout2d(0.25),
            custom2DCNN(80, 160, 2, "same"),
            nn.MaxPool2d((2, 1)), # 16, 5
            nn.Dropout2d(0.25),
            custom2DCNN(160, 160, 2, "same"),
            nn.MaxPool2d(2), # 8, 2
            nn.Dropout2d(0.25),
            custom2DCNN(160, 160, 2, "same"),
            nn.MaxPool2d((2, 1)), #4, 2
            nn.Dropout2d(0.25),
            custom2DCNN(160, 160, 2, "same"),
            nn.MaxPool2d((4, 2)),
            nn.Dropout2d(0.25),
        )

    def forward(self, x):
        x_harm, x_perc = self.hpss(x)
        # x1, x2, x3 = torch.split(self.logmel(x_harm), 70, dim=2)
        # x4, x5, x6 = torch.split(self.logmel(x_perc), 70, dim=2)
        x1 = self.logmel(x_harm)
        x2 = self.logmel(x_perc)

        x_env = self.env(x)
        x_env = x_env[:, :, :-1]
        x_env = self.cnn_env(x_env)

        x1 = self.cnn1(x1) 
        x2 = self.cnn2(x2)
        # x3 = self.cnn3(x3)
        # x4 = self.cnn4(x4)
        # x5 = self.cnn5(x5)
        # x6 = self.cnn6(x6)

        x = torch.cat((x1, x2, x_env.unsqueeze(3)), dim=1)

        x_flat = x.view(x.size(0), -1)
        z = self.fc(x_flat)
        return z
    
class v1_mi6_hpss_only(nn.Module):
    def __init__(self, output_nbr, args):
        super(v1_mi6_hpss_only, self).__init__()

        self.sr = args.sr
        self.device = args.device

        self.logmel = LogMelSpectrogramLayer(sample_rate=self.sr, n_mels=128, n_fft=2048, hop_length=512)
        self.hpss = HPSS(n_fft=2048, hop_length=512, kernel_size=31, device=self.device)
        
        self.cnn1 = self._create_cnn_block()
        self.cnn2 = self._create_cnn_block()

        self.fc = nn.Sequential(
            nn.Linear(160 * 2, 160),
            nn.ReLU(),
            nn.Linear(160, 40),
            nn.ReLU(),
            nn.Linear(40, output_nbr)
        )

    def _create_cnn_block(self):
        return nn.Sequential(
            custom2DCNN(1, 40, (2, 3), "same"),
            custom2DCNN(40, 40, (2, 3), "same"),
            nn.MaxPool2d((2, 1)), # 64, 15
            nn.Dropout2d(0.25),
            custom2DCNN(40, 80, (2, 3), "same"),
            custom2DCNN(80, 80, (2, 3), "same"),
            nn.MaxPool2d((2, 3)), # 32, 5
            nn.Dropout2d(0.25),
            custom2DCNN(80, 160, 2, "same"),
            nn.MaxPool2d((2, 1)), # 16, 5
            nn.Dropout2d(0.25),
            custom2DCNN(160, 160, 2, "same"),
            nn.MaxPool2d(2), # 8, 2
            nn.Dropout2d(0.25),
            custom2DCNN(160, 160, 2, "same"),
            nn.MaxPool2d((2, 1)), #4, 2
            nn.Dropout2d(0.25),
            custom2DCNN(160, 160, 2, "same"),
            nn.MaxPool2d((4, 2)),
            nn.Dropout2d(0.25),
        )

    def forward(self, x):
        x_harm, x_perc = self.hpss(x)
        x1 = self.logmel(x_harm)
        x2 = self.logmel(x_perc)

        x1 = self.cnn1(x1) 
        x2 = self.cnn2(x2)

        x = torch.cat((x1, x2), dim=1)

        x_flat = x.view(x.size(0), -1)
        z = self.fc(x_flat)
        return z

class v1_mi6_env2_perc(nn.Module):
    def __init__(self, output_nbr, args):
        super(v1_mi6_env2_perc, self).__init__()

        self.sr = args.sr

        self.logmel = LogMelSpectrogramLayer(sample_rate=self.sr, n_mels=420)
        self.env = EnvelopeFollowingLayerTorchScript(n_fft=2048, hop_length=512, smoothing_factor=4)
        self.hpss = HPSS(n_fft=2048, hop_length=512, kernel_size=31, device=self.device)
        
        self.cnn1 = self._create_cnn_block()
        self.cnn2 = self._create_cnn_block()
        self.cnn3 = self._create_cnn_block()
        self.cnn4 = self._create_cnn_block()
        self.cnn5 = self._create_cnn_block()
        self.cnn6 = self._create_cnn_block()

        self.cnn_env = self._create_cnn_env_block()

        self.cnn_hpss = self._create_cnn_block()

        self.fc = nn.Sequential(
                nn.Linear(160 * 8, 560),
                nn.BatchNorm1d(560),
                nn.ReLU(),
                nn.Dropout1d(0.25),
                nn.Linear(560, 280),
                nn.BatchNorm1d(280),
                nn.ReLU(),
                nn.Dropout1d(0.25),
                nn.Linear(280, 140),
                nn.BatchNorm1d(140),
                nn.ReLU(),
                nn.Dropout1d(0.25),
                nn.Linear(140, 70),
                nn.BatchNorm1d(70),
                nn.ReLU(),
                nn.Dropout1d(0.25),
                nn.Linear(70, output_nbr)
            )

    def _create_cnn_env_block(self):
        return nn.Sequential(
            custom1DCNN(1, 40, 7, "same", 4),
            nn.AvgPool1d(16),
            custom1DCNN(40, 40, 5, "same", 3),
            nn.AvgPool1d(8),
            custom1DCNN(40, 80, 3, "same", 2),
            nn.AvgPool1d(8),
            custom1DCNN(80, 160, 2, "same", 1),
            nn.AvgPool1d(7),
            nn.Dropout1d(0.1),
        )

    def _create_cnn_block(self):
        return nn.Sequential(
            custom2DCNN(1, 40, (2, 3), "same"),
            custom2DCNN(40, 40, (2, 3), "same"),
            nn.MaxPool2d((2, 1)), # 35
            nn.Dropout2d(0.25),
            custom2DCNN(40, 80, (2, 3), "same"),
            custom2DCNN(80, 80, (2, 3), "same"),
            nn.MaxPool2d((2, 3)), # 17
            nn.Dropout2d(0.25),
            custom2DCNN(80, 160, 2, "same"),
            nn.MaxPool2d((2, 1)), # 8
            nn.Dropout2d(0.25),
            custom2DCNN(160, 160, 2, "same"),
            nn.MaxPool2d(2), # 4
            nn.Dropout2d(0.25),
            custom2DCNN(160, 160, 2, "same"),
            nn.MaxPool2d((2, 1)), #2
            nn.Dropout2d(0.25),
            custom2DCNN(160, 160, 2, "same"),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.25),
        )

    def forward(self, x):
        x_env = self.env(x)
        x_env = x_env[:, :, :-1]
        x_env = self.cnn_env(x_env)

        _, x_perc = self.hpss(x)
        x_perc = self.cnn_hpss(x_perc)

        x1, x2, x3, x4, x5, x6 = torch.split(self.logmel(x), 70, dim=2)

        x1 = self.cnn1(x1) 
        x2 = self.cnn2(x2)
        x3 = self.cnn3(x3)
        x4 = self.cnn4(x4)
        x5 = self.cnn5(x5)
        x6 = self.cnn6(x6)

        x = torch.cat((x1, x2, x3, x4, x5, x6, x_env.unsqueeze(3), x_perc), dim=1)

        x_flat = x.view(x.size(0), -1)
        z = self.fc(x_flat)
        return z



class v1_mi6_env2_mod_new(nn.Module):
    def __init__(self, output_nbr, args):
        super(v1_mi6_env2_mod_new, self).__init__()

        self.sr = args.sr

        # self.logmel1 = LogMelSpectrogramLayer(sample_rate=self.sr, n_mels=420, n_fft=2048, hop_length=512)
        # self.logmel2 = LogMelSpectrogramLayer(sample_rate=self.sr, n_mels=420, n_fft=2048, hop_length=256)
        self.logmel3 = LogMelSpectrogramLayer(sample_rate=self.sr, n_mels=420, n_fft=2048, hop_length=128)
        self.env = EnvelopeFollowingLayerTorchScript(n_fft=2048, hop_length=128, smoothing_factor=128)
        
        self.cnn1 = self._create_cnn_block()
        self.cnn2 = self._create_cnn_block()
        self.cnn3 = self._create_cnn_block()
        self.cnn4 = self._create_cnn_block()
        self.cnn5 = self._create_cnn_block()
        self.cnn6 = self._create_cnn_block()

        self.cnn_env = self._create_cnn_env_block()

        self.fc = nn.Sequential(
                nn.Linear(160 * 7, 320),
                nn.BatchNorm1d(320),
                nn.ReLU(),
                nn.Dropout1d(0.25),
                nn.Linear(320, 80),
                nn.BatchNorm1d(80),
                nn.ReLU(),
                nn.Dropout1d(0.25),
                nn.Linear(80, output_nbr)
            )

    def _create_cnn_env_block(self):
        # change dilation
        return nn.Sequential(
            custom1DCNN(1, 40, 7, "same", 1),
            nn.AvgPool1d(16),
            nn.Dropout1d(0.25),
            custom1DCNN(40, 40, 5, "same", 2),
            nn.AvgPool1d(8),
            nn.Dropout1d(0.25),
            custom1DCNN(40, 80, 3, "same", 3),
            nn.AvgPool1d(8),
            nn.Dropout1d(0.25),
            custom1DCNN(80, 160, 2, "same", 4),
            nn.AvgPool1d(7),
            nn.Dropout1d(0.25),
        )

    def _create_cnn_block(self):
        return nn.Sequential(
            # 70 * 57
            custom2DCNN(1, 40, (3, 7), "same"),
            custom2DCNN(40, 40, (3, 7), "same"),
            nn.MaxPool2d(2), # 35, 28
            nn.Dropout2d(0.25),
            custom2DCNN(40, 80, (3, 5), "same"),
            custom2DCNN(80, 80, (3, 5), "same"),
            nn.MaxPool2d(2), # 17, 14
            nn.Dropout2d(0.25),
            custom2DCNN(80, 160, (2, 4), "same"),
            custom2DCNN(160, 160, (2, 4), "same"),
            nn.MaxPool2d(2), # 8, 7
            nn.Dropout2d(0.25),
            custom2DCNN(160, 160, (2, 3), "same"),
            nn.MaxPool2d(2), #4, 3
            nn.Dropout2d(0.25),
            custom2DCNN(160, 160, (1, 2), "same"), 
            nn.MaxPool2d(2), #2, 1
            nn.Dropout2d(0.25),
            custom2DCNN(160, 160, (2, 1), "same"),
            nn.MaxPool2d((2, 1)),
            nn.Dropout2d(0.25),
        )

    def forward(self, x):
        x_env = self.env(x)
        x_env = x_env[:, :, :-1]
        x_env = self.cnn_env(x_env)

        x4_1, x4_2, x4_3, x4_4, x4_5, x4_6 = torch.split(self.logmel3(x), 70, dim=2)

        x1 = self.cnn1(x4_1) 
        x2 = self.cnn2(x4_2)
        x3 = self.cnn3(x4_3)
        x4 = self.cnn4(x4_4)
        x5 = self.cnn5(x4_5)
        x6 = self.cnn6(x4_6)

        x = torch.cat((x1, x2, x3, x4, x5, x6, x_env.unsqueeze(3)), dim=1)

        x_flat = x.view(x.size(0), -1)
        z = self.fc(x_flat)
        return z

# GOOD
# class v1_mi6_env2_mod_stacks7(nn.Module):
#     def __init__(self, output_nbr, args):
#         super(v1_mi6_env2_mod_stacks7, self).__init__()

#         self.sr = args.sr

#         self.logmel1 = LogMelSpectrogramLayer(sample_rate=self.sr, n_mels=420, n_fft=2048, hop_length=512)
#         self.logmel2 = LogMelSpectrogramLayer(sample_rate=self.sr, n_mels=420, n_fft=2048, hop_length=256)
#         self.logmel3 = LogMelSpectrogramLayer(sample_rate=self.sr, n_mels=420, n_fft=2048, hop_length=128)
#         self.env = EnvelopeFollowingLayerTorchScript(n_fft=2048, hop_length=512, smoothing_factor=4)
        
#         self.cnn1 = self._create_cnn_block()
#         self.cnn2 = self._create_cnn_block()
#         self.cnn3 = self._create_cnn_block()
#         self.cnn4 = self._create_cnn_block()
#         self.cnn5 = self._create_cnn_block()
#         self.cnn6 = self._create_cnn_block()

#         self.cnn_env = self._create_cnn_env_block()

#         self.fc = nn.Sequential(
#                 nn.Linear(160 * 7, 320),
#                 nn.BatchNorm1d(320),
#                 nn.ReLU(),
#                 nn.Dropout1d(0.25),
#                 nn.Linear(320, 80),
#                 nn.BatchNorm1d(80),
#                 nn.ReLU(),
#                 nn.Dropout1d(0.25),
#                 nn.Linear(80, output_nbr)
#             )

#     def _create_cnn_env_block(self):
#         # change dilation
#         return nn.Sequential(
#             custom1DCNN(1, 40, 7, "same", 2),
#             nn.AvgPool1d(16),
#             nn.Dropout1d(0.25),
#             custom1DCNN(40, 40, 5, "same", 3),
#             nn.AvgPool1d(8),
#             nn.Dropout1d(0.25),
#             custom1DCNN(40, 80, 3, "same", 4),
#             nn.AvgPool1d(8),
#             nn.Dropout1d(0.25),
#             custom1DCNN(80, 160, 2, "same", 5),
#             nn.AvgPool1d(7),
#             nn.Dropout1d(0.25),
#         )

#     def _create_cnn_block(self):
#         return nn.Sequential(
#             custom2DCNN(7, 40, (2, 7), "same"), # old (2, 7)
#             custom2DCNN(40, 40, (2, 7), "same"), # old (2, 7)
#             nn.MaxPool2d((2, 1)), # 35, 15
#             nn.Dropout2d(0.25),
#             custom2DCNN(40, 80, (2, 5), "same"),
#             custom2DCNN(80, 80, (2, 5), "same"),
#             nn.MaxPool2d((2, 3)), # 17, 5
#             nn.Dropout2d(0.25),
#             custom2DCNN(80, 160, (2, 3), "same"),
#             nn.MaxPool2d((2, 1)), # 8, 5
#             nn.Dropout2d(0.25),
#             custom2DCNN(160, 160, (2, 3), "same"),
#             nn.MaxPool2d(2), # 4, 2
#             nn.Dropout2d(0.25),
#             custom2DCNN(160, 160, (1, 2), "same"),
#             nn.MaxPool2d((2, 1)), #2, 2
#             nn.Dropout2d(0.25),
#             custom2DCNN(160, 160, (1, 2), "same"), # old 2
#             nn.MaxPool2d(2),
#             nn.Dropout2d(0.25),
#         )

#     def forward(self, x):
#         x_env = self.env(x)
#         x_env = x_env[:, :, :-1]
#         x_env = self.cnn_env(x_env)

#         x1_1, x1_2, x1_3, x1_4, x1_5, x1_6 = torch.split(self.logmel1(x), 70, dim=2)
#         x2_1, x2_2, x2_3, x2_4, x2_5, x2_6 = torch.split(self.logmel2(x)[:,:,:, :15], 70, dim=2)
#         x3_1, x3_2, x3_3, x3_4, x3_5, x3_6 = torch.split(self.logmel2(x)[:,:,:, 14:], 70, dim=2)

#         x4_1, x4_2, x4_3, x4_4, x4_5, x4_6 = torch.split(self.logmel3(x)[:,:,:, :15], 70, dim=2)
#         x5_1, x5_2, x5_3, x5_4, x5_5, x5_6 = torch.split(self.logmel3(x)[:,:,:, 14:29], 70, dim=2)
#         x6_1, x6_2, x6_3, x6_4, x6_5, x6_6 = torch.split(self.logmel3(x)[:,:,:, 28:43], 70, dim=2)
#         x7_1, x7_2, x7_3, x7_4, x7_5, x7_6 = torch.split(self.logmel3(x)[:,:,:, 42:], 70, dim=2)
        
#         rx1_s = torch.cat((x1_1, x2_1, x3_1, x4_1, x5_1, x6_1, x7_1), dim=1)
#         rx2_s = torch.cat((x1_2, x2_2, x3_2, x4_2, x5_2, x6_2, x7_2), dim=1)
#         rx3_s = torch.cat((x1_3, x2_3, x3_3, x4_3, x5_3, x6_3, x7_3), dim=1)
#         rx4_s = torch.cat((x1_4, x2_4, x3_4, x4_4, x5_4, x6_4, x7_4), dim=1)
#         rx5_s = torch.cat((x1_5, x2_5, x3_5, x4_5, x5_5, x6_5, x7_5), dim=1)
#         rx6_s = torch.cat((x1_6, x2_6, x3_6, x4_6, x5_6, x6_6, x7_6), dim=1)

#         x1 = self.cnn1(rx1_s) 
#         x2 = self.cnn2(rx2_s)
#         x3 = self.cnn3(rx3_s)
#         x4 = self.cnn4(rx4_s)
#         x5 = self.cnn5(rx5_s)
#         x6 = self.cnn6(rx6_s)

#         x = torch.cat((x1, x2, x3, x4, x5, x6, x_env.unsqueeze(3)), dim=1)

#         x_flat = x.view(x.size(0), -1)
#         z = self.fc(x_flat)
#         return z
    
class v1_mi6_mod_stacks7(nn.Module):
    def __init__(self, output_nbr, args):
        super(v1_mi6_mod_stacks7, self).__init__()

        self.sr = args.sr

        self.logmel1 = LogMelSpectrogramLayer(sample_rate=self.sr, n_mels=420, n_fft=2048, hop_length=512)
        self.logmel2 = LogMelSpectrogramLayer(sample_rate=self.sr, n_mels=420, n_fft=2048, hop_length=256)
        self.logmel3 = LogMelSpectrogramLayer(sample_rate=self.sr, n_mels=420, n_fft=2048, hop_length=128)
        
        self.cnn1 = self._create_cnn_block()
        self.cnn2 = self._create_cnn_block()
        self.cnn3 = self._create_cnn_block()
        self.cnn4 = self._create_cnn_block()
        self.cnn5 = self._create_cnn_block()
        self.cnn6 = self._create_cnn_block()

        self.fc = nn.Sequential(
                nn.Linear(160 * 6, 320),
                nn.BatchNorm1d(320),
                nn.ReLU(),
                nn.Dropout1d(0.25),
                nn.Linear(320, 80),
                nn.BatchNorm1d(80),
                nn.ReLU(),
                nn.Dropout1d(0.25),
                nn.Linear(80, output_nbr)
            )

    def _create_cnn_block(self):
        return nn.Sequential(
            custom2DCNN(7, 40, (2, 7), "same"),
            custom2DCNN(40, 40, (2, 7), "same"),
            nn.MaxPool2d((2, 1)), # 35, 15
            nn.Dropout2d(0.25),
            custom2DCNN(40, 80, (2, 5), "same"),
            custom2DCNN(80, 80, (2, 5), "same"),
            nn.MaxPool2d((2, 3)), # 17, 5
            nn.Dropout2d(0.25),
            custom2DCNN(80, 160, (2, 3), "same"),
            nn.MaxPool2d((2, 1)), # 8, 5
            nn.Dropout2d(0.25),
            custom2DCNN(160, 160, (1, 3), "same"),
            nn.MaxPool2d(2), # 4, 2
            nn.Dropout2d(0.25),
            custom2DCNN(160, 160, (1, 2), "same"),
            nn.MaxPool2d((2, 1)), #2, 2
            nn.Dropout2d(0.25),
            # custom2DCNN(160, 160, 2, "same"),
            custom2DCNN(160, 160, (1, 2), "same"),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.25),
        )

    def forward(self, x):

        x1_1, x1_2, x1_3, x1_4, x1_5, x1_6 = torch.split(self.logmel1(x), 70, dim=2)

        x2_1, x2_2, x2_3, x2_4, x2_5, x2_6 = torch.split(self.logmel2(x)[:,:,:, :15], 70, dim=2)
        x3_1, x3_2, x3_3, x3_4, x3_5, x3_6 = torch.split(self.logmel2(x)[:,:,:, 14:], 70, dim=2)

        x4_1, x4_2, x4_3, x4_4, x4_5, x4_6 = torch.split(self.logmel3(x)[:,:,:, :15], 70, dim=2)
        x5_1, x5_2, x5_3, x5_4, x5_5, x5_6 = torch.split(self.logmel3(x)[:,:,:, 14:29], 70, dim=2)
        x6_1, x6_2, x6_3, x6_4, x6_5, x6_6 = torch.split(self.logmel3(x)[:,:,:, 28:43], 70, dim=2)
        x7_1, x7_2, x7_3, x7_4, x7_5, x7_6 = torch.split(self.logmel3(x)[:,:,:, 42:], 70, dim=2)
        
        rx1_s = torch.cat((x1_1, x2_1, x3_1, x4_1, x5_1, x6_1, x7_1), dim=1)
        rx2_s = torch.cat((x1_2, x2_2, x3_2, x4_2, x5_2, x6_2, x7_2), dim=1)
        rx3_s = torch.cat((x1_3, x2_3, x3_3, x4_3, x5_3, x6_3, x7_3), dim=1)
        rx4_s = torch.cat((x1_4, x2_4, x3_4, x4_4, x5_4, x6_4, x7_4), dim=1)
        rx5_s = torch.cat((x1_5, x2_5, x3_5, x4_5, x5_5, x6_5, x7_5), dim=1)
        rx6_s = torch.cat((x1_6, x2_6, x3_6, x4_6, x5_6, x6_6, x7_6), dim=1)

        x1 = self.cnn1(rx1_s) 
        x2 = self.cnn2(rx2_s)
        x3 = self.cnn3(rx3_s)
        x4 = self.cnn4(rx4_s)
        x5 = self.cnn5(rx5_s)
        x6 = self.cnn6(rx6_s)

        x = torch.cat((x1, x2, x3, x4, x5, x6), dim=1)

        x_flat = x.view(x.size(0), -1)
        z = self.fc(x_flat)
        return z
    

class v1_mi6_env2_mod_full_stack(nn.Module):
    def __init__(self, output_nbr, args):
        super(v1_mi6_env2_mod_full_stack, self).__init__()

        self.sr = args.sr

        self.logmel1 = LogMelSpectrogramLayer(sample_rate=self.sr, n_mels=420, n_fft=2048, hop_length=512)
        self.logmel2 = LogMelSpectrogramLayer(sample_rate=self.sr, n_mels=420, n_fft=2048, hop_length=256)
        self.logmel3 = LogMelSpectrogramLayer(sample_rate=self.sr, n_mels=420, n_fft=2048, hop_length=128)

        self.env1 = EnvelopeFollowingLayerTorchScript(n_fft=2048, hop_length=512, smoothing_factor=128)
        self.env2 = EnvelopeFollowingLayerTorchScript(n_fft=2048, hop_length=512, smoothing_factor=256)
        self.env3 = EnvelopeFollowingLayerTorchScript(n_fft=2048, hop_length=512, smoothing_factor=512)

        
        self.cnn1 = self._create_cnn_block()
        self.cnn2 = self._create_cnn_block()
        self.cnn3 = self._create_cnn_block()
        self.cnn4 = self._create_cnn_block()
        self.cnn5 = self._create_cnn_block()
        self.cnn6 = self._create_cnn_block()

        self.cnn_env = self._create_cnn_env_block()

        self.fc = nn.Sequential(
                nn.Linear(160 * 7, 320),
                nn.BatchNorm1d(320),
                nn.ReLU(),
                nn.Dropout1d(0.25),
                nn.Linear(320, 80),
                nn.BatchNorm1d(80),
                nn.ReLU(),
                nn.Dropout1d(0.25),
                nn.Linear(80, output_nbr)
            )

    def _create_cnn_env_block(self):
        # change dilation
        return nn.Sequential(
            custom1DCNN(3, 40, 7, "same", 2),
            nn.AvgPool1d(16),
            nn.Dropout1d(0.25),
            custom1DCNN(40, 40, 5, "same", 3),
            nn.AvgPool1d(8),
            nn.Dropout1d(0.25),
            custom1DCNN(40, 80, 3, "same", 4),
            nn.AvgPool1d(8),
            nn.Dropout1d(0.25),
            custom1DCNN(80, 160, 2, "same", 5),
            nn.AvgPool1d(7),
            nn.Dropout1d(0.25),
        )

    def _create_cnn_block(self):
        return nn.Sequential(
            custom2DCNN(7, 40, (2, 7), "same"), # old (2,7) -> revenir à 3, 7
            custom2DCNN(40, 40, (2, 7), "same"), # old (2, 7) -> revenir à 3, 7
            nn.MaxPool2d((2, 1)), # 35, 15
            nn.Dropout2d(0.25),
            custom2DCNN(40, 80, (2, 5), "same"),
            custom2DCNN(80, 80, (2, 5), "same"),
            nn.MaxPool2d((2, 3)), # 17, 5
            nn.Dropout2d(0.25),
            custom2DCNN(80, 160, (2, 3), "same"),
            nn.MaxPool2d((2, 1)), # 8, 5
            nn.Dropout2d(0.25),
            custom2DCNN(160, 160, (2, 3), "same"),
            nn.MaxPool2d(2), # 4, 2
            nn.Dropout2d(0.25),
            custom2DCNN(160, 160, (1, 2), "same"),
            nn.MaxPool2d((2, 1)), #2, 2
            nn.Dropout2d(0.25),
            custom2DCNN(160, 160, (1, 2), "same"), # old 2
            nn.MaxPool2d(2),
            nn.Dropout2d(0.25),
        )

    def forward(self, x):
        x_env1 = self.env1(x)[:, :, :-1]
        x_env2 = self.env2(x)[:, :, :-1]
        x_env3 = self.env3(x)[:, :, :-1]
        x_env = torch.cat((x_env1, x_env2, x_env3), dim=1)
        x_env = self.cnn_env(x_env)

        x1_1, x1_2, x1_3, x1_4, x1_5, x1_6 = torch.split(self.logmel1(x), 70, dim=2)
        x2_1, x2_2, x2_3, x2_4, x2_5, x2_6 = torch.split(self.logmel2(x)[:,:,:, :15], 70, dim=2)
        x3_1, x3_2, x3_3, x3_4, x3_5, x3_6 = torch.split(self.logmel2(x)[:,:,:, 14:], 70, dim=2)

        x4_1, x4_2, x4_3, x4_4, x4_5, x4_6 = torch.split(self.logmel3(x)[:,:,:, :15], 70, dim=2)
        x5_1, x5_2, x5_3, x5_4, x5_5, x5_6 = torch.split(self.logmel3(x)[:,:,:, 14:29], 70, dim=2)
        x6_1, x6_2, x6_3, x6_4, x6_5, x6_6 = torch.split(self.logmel3(x)[:,:,:, 28:43], 70, dim=2)
        x7_1, x7_2, x7_3, x7_4, x7_5, x7_6 = torch.split(self.logmel3(x)[:,:,:, 42:], 70, dim=2)
        
        rx1_s = torch.cat((x1_1, x2_1, x3_1, x4_1, x5_1, x6_1, x7_1), dim=1)
        rx2_s = torch.cat((x1_2, x2_2, x3_2, x4_2, x5_2, x6_2, x7_2), dim=1)
        rx3_s = torch.cat((x1_3, x2_3, x3_3, x4_3, x5_3, x6_3, x7_3), dim=1)
        rx4_s = torch.cat((x1_4, x2_4, x3_4, x4_4, x5_4, x6_4, x7_4), dim=1)
        rx5_s = torch.cat((x1_5, x2_5, x3_5, x4_5, x5_5, x6_5, x7_5), dim=1)
        rx6_s = torch.cat((x1_6, x2_6, x3_6, x4_6, x5_6, x6_6, x7_6), dim=1)

        x1 = self.cnn1(rx1_s) 
        x2 = self.cnn2(rx2_s)
        x3 = self.cnn3(rx3_s)
        x4 = self.cnn4(rx4_s)
        x5 = self.cnn5(rx5_s)
        x6 = self.cnn6(rx6_s)

        x = torch.cat((x1, x2, x3, x4, x5, x6, x_env.unsqueeze(3)), dim=1)

        x_flat = x.view(x.size(0), -1)
        z = self.fc(x_flat)
        return z
    
class context_net(nn.Module):
    def __init__(self, output_nbr, args):
        super(context_net, self).__init__()

        self.sr = args.sr

        self.logmel1 = LogMelSpectrogramLayer(sample_rate=self.sr, n_mels=420, n_fft=2048, hop_length=512)
        self.logmel2 = LogMelSpectrogramLayer(sample_rate=self.sr, n_mels=420, n_fft=2048, hop_length=256)

        self.env = EnvelopeFollowingLayerTorchScript(n_fft=2048, hop_length=512, smoothing_factor=4)
        
        self.cnn1_context = self._create_cnn_block()
        self.cnn2_context = self._create_cnn_block()
        self.cnn3_context = self._create_cnn_block()
        self.cnn4_context = self._create_cnn_block()

        self.cnn5_local = self._create_cnn_block()
        self.cnn6_local = self._create_cnn_block()
        self.cnn7_local = self._create_cnn_block()
        self.cnn8_local = self._create_cnn_block()

        self.cnn9_local = self._create_cnn_block()
        self.cnn10_local = self._create_cnn_block()
        self.cnn11_local = self._create_cnn_block()
        self.cnn12_local = self._create_cnn_block()

        self.merge1 = self._create_merge_block1()
        self.merge2 = self._create_merge_block1()
        self.merge3 = self._create_merge_block1()
        self.merge4 = self._create_merge_block1()

        self.merge5 = self._create_merge_block1()
        self.merge6 = self._create_merge_block1()
        self.merge7 = self._create_merge_block1()
        self.merge8 = self._create_merge_block1()

        self.w1_merge = self._create_merge_block2()

        self.cnn_env = self._create_cnn_env_block()

        self.fc = nn.Sequential(
                nn.Linear(160 * 3, 160),
                nn.BatchNorm1d(160),
                nn.ReLU(),
                nn.Dropout1d(0.25),
                nn.Linear(160, 80),
                nn.BatchNorm1d(80),
                nn.ReLU(),
                nn.Dropout1d(0.25),
                nn.Linear(80, output_nbr)
            )

    def _create_merge_block1(self):
        return nn.Sequential(
            nn.Linear(80 * 2, 80),
            nn.BatchNorm1d(80),
            nn.ReLU(),
        )

    def _create_merge_block2(self):
        return nn.Sequential(
            nn.Linear(80 * 4, 160),
            nn.BatchNorm1d(160),
            nn.ReLU(),
        )

    def _create_cnn_env_block(self):
        # change dilation
        return nn.Sequential(
            custom1DCNN(1, 40, 7, "same", 2),
            nn.AvgPool1d(16),
            nn.Dropout1d(0.25),
            custom1DCNN(40, 40, 5, "same", 3),
            nn.AvgPool1d(8),
            nn.Dropout1d(0.25),
            custom1DCNN(40, 80, 3, "same", 4),
            nn.AvgPool1d(8),
            nn.Dropout1d(0.25),
            custom1DCNN(80, 160, 2, "same", 5),
            nn.AvgPool1d(7),
            nn.Dropout1d(0.25),
        )

    def _create_cnn_block(self):
        return nn.Sequential(
            custom2DCNN(1, 20, (3, 7), "same"), # old (2,7) -> revenir à 3, 7
            custom2DCNN(20, 20, (3, 7), "same"), # old (2, 7) -> revenir à 3, 7
            nn.MaxPool2d((2, 1)), # 52, 15
            nn.Dropout2d(0.25),
            custom2DCNN(20, 40, (2, 5), "same"),
            custom2DCNN(40, 40, (2, 5), "same"),
            nn.MaxPool2d((2, 3)), # 26, 5
            nn.Dropout2d(0.25),
            custom2DCNN(40, 80, (2, 3), "same"),
            nn.MaxPool2d((2, 1)), # 13, 5
            nn.Dropout2d(0.25),
            custom2DCNN(80, 80, (2, 3), "same"),
            nn.MaxPool2d(2), # 6, 2
            nn.Dropout2d(0.25),
            custom2DCNN(80, 80, (1, 2), "same"),
            nn.MaxPool2d((2, 1)), #3, 2
            nn.Dropout2d(0.25),
            custom2DCNN(80, 80, (1, 2), "same"), # old 2
            nn.MaxPool2d((3, 2)),
            nn.Dropout2d(0.25),
        )

    def forward(self, x):
        x_env = self.env(x)[:, :, :-1]
        x_env = self.cnn_env(x_env)

        x1_1, x1_2, x1_3, x1_4 = torch.split(self.logmel1(x), 105, dim=2)

        x2_1, x2_2, x2_3, x2_4 = torch.split(self.logmel2(x)[:,:,:, :15], 105, dim=2)
        x3_1, x3_2, x3_3, x3_4 = torch.split(self.logmel2(x)[:,:,:, 14:], 105, dim=2)
        
        c1 = self.cnn1_context(x1_1)
        c2 = self.cnn2_context(x1_2)
        c3 = self.cnn3_context(x1_3)
        c4 = self.cnn4_context(x1_4)
        
        l1 = self.cnn5_local(x2_1)
        l2 = self.cnn6_local(x2_2)
        l3 = self.cnn7_local(x2_3)
        l4 = self.cnn8_local(x2_4)

        l5 = self.cnn9_local(x3_1)
        l6 = self.cnn10_local(x3_2)
        l7 = self.cnn11_local(x3_3)
        l8 = self.cnn12_local(x3_4)

        x1 = torch.cat((c1, l1), dim=1)
        x2 = torch.cat((c2, l2), dim=1)
        x3 = torch.cat((c3, l3), dim=1)
        x4 = torch.cat((c4, l4), dim=1)

        x5 = torch.cat((c1, l5), dim=1)
        x6 = torch.cat((c2, l6), dim=1)
        x7 = torch.cat((c3, l7), dim=1)
        x8 = torch.cat((c4, l8), dim=1)

        m1 = self.merge1(x1)
        m2 = self.merge2(x2)
        m3 = self.merge3(x3)
        m4 = self.merge4(x4)

        m5 = self.merge5(x5)
        m6 = self.merge6(x6)
        m7 = self.merge7(x7)
        m8 = self.merge8(x8)

        w1 = torch.cat((m1, m2, m3, m4), dim=1)
        w2 = torch.cat((m5, m6, m7, m8), dim=1)
        w1 = self.w1_merge(w1)
        w2 = self.w1_merge(w2)

        z = torch.cat((w1, w2, x_env.squeeze(3)), dim=1)
        z = self.fc(z)

        return z