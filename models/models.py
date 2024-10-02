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

from models.layers import LogMelSpectrogramLayer, custom2DCNN, custom1DCNN, EnvelopeFollowingLayerTorchScript
from utils.constants import SEGMENT_LENGTH

class v1(nn.Module):
    def __init__(self, output_nbr, sr):
        super(v1, self).__init__()

        self.logmel = LogMelSpectrogramLayer(sample_rate=sr)
        
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
    def __init__(self, output_nbr, sr):
        super(v1_mi6_env2, self).__init__()

        self.logmel = LogMelSpectrogramLayer(sample_rate=sr, n_mels=420, hop_length=512)
        self.env = EnvelopeFollowingLayerTorchScript(n_fft=2048, hop_length=512, smoothing_factor=200)
        
        self.cnn1 = self._create_cnn_block()
        self.cnn2 = self._create_cnn_block()
        self.cnn3 = self._create_cnn_block()
        self.cnn4 = self._create_cnn_block()
        self.cnn5 = self._create_cnn_block()
        self.cnn6 = self._create_cnn_block()

        self.cnn_env = self._create_cnn_env_block()

        self.fc = nn.Sequential(
            nn.Linear(160 * 7, 320),
            nn.ReLU(),
            nn.Linear(320, 80),
            nn.ReLU(),
            nn.Linear(80, output_nbr)
        )

    def _create_cnn_env_block(self):
        return nn.Sequential(
            nn.BatchNorm1d(1),
            custom1DCNN(1, 40, 7, "same", 4),
            nn.AvgPool1d(16),
            custom1DCNN(40, 40, 5, "same", 3),
            nn.AvgPool1d(16),
            custom1DCNN(40, 80, 3, "same", 2),
            nn.AvgPool1d(7),
            custom1DCNN(80, 160, 2, "same", 1),
            nn.AvgPool1d(4),
            nn.Dropout1d(0.25),
        )

    def _create_cnn_block(self):
        return nn.Sequential(
            custom2DCNN(1, 40, (2, 3), "same"),
            custom2DCNN(40, 40, (2, 3), "same"),
            nn.MaxPool2d((2, 1)), # 35, 15
            nn.Dropout2d(0.25),
            custom2DCNN(40, 80, (2, 3), "same"),
            custom2DCNN(80, 80, (2, 3), "same"),
            nn.MaxPool2d((2, 3)), # 17, 5
            nn.Dropout2d(0.25),
            custom2DCNN(80, 160, 2, "same"),
            nn.MaxPool2d(2), # 8, 2
            nn.Dropout2d(0.25),
            custom2DCNN(160, 160, 2, "same"),
            nn.MaxPool2d((2, 1)), # 4, 2
            nn.Dropout2d(0.25),
            custom2DCNN(160, 160, 2, "same"),
            nn.MaxPool2d((2, 1)), #2, 2
            nn.Dropout2d(0.25),
            custom2DCNN(160, 160, 2, "same"),
            nn.MaxPool2d(2), # 
            nn.Dropout2d(0.25),
        )

    def forward(self, x):
        x_env = self.env(x)
        # x_env = x_env[:, :, :-1]
        # print(x_env.shape)
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
    
class v1_mi6_env2_lstm(nn.Module):
    def __init__(self, output_nbr, sr):
        super(v1_mi6_env2_lstm, self).__init__()

        self.logmel = LogMelSpectrogramLayer(sample_rate=sr, n_mels=384, hop_length=512)
        self.env = EnvelopeFollowingLayerTorchScript(n_fft=2048, hop_length=512, smoothing_factor=4)

        self.relu = nn.ReLU()
        
        self.cnn1 = self._create_cnn_block()
        self.cnn2 = self._create_cnn_block()
        self.cnn3 = self._create_cnn_block()
        self.cnn4 = self._create_cnn_block()
        self.cnn5 = self._create_cnn_block()
        self.cnn6 = self._create_cnn_block()

        self.cnn_env = self._create_cnn_env_block()
        self.lstm_env = self._create_lstm_env_block()

        self.fc = nn.Sequential(
            nn.Linear(128 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, 64), 
            nn.ReLU(),
            nn.Linear(64, output_nbr)
        )

    def _create_cnn_env_block(self):
        return nn.Sequential(
            custom1DCNN(1, 32, 7, "same", 4),
            nn.AvgPool1d(8),
            custom1DCNN(32, 64, 5, "same", 3),
            nn.AvgPool1d(7),
            custom1DCNN(64, 128, 2, "same", 1),
        )
    
    def _create_lstm_env_block(self):
        return nn.LSTM(input_size=128, hidden_size=128, batch_first=True)

    def _create_cnn_block(self):
        return nn.Sequential(
            custom2DCNN(1, 32, (2, 3), "same"),
            custom2DCNN(32, 32, (2, 3), "same"),
            nn.MaxPool2d((2, 1)), # 32, 15
            nn.Dropout2d(0.25),
            custom2DCNN(32, 64, (2, 3), "same"),
            custom2DCNN(64, 64, (2, 3), "same"),
            nn.MaxPool2d((2, 3)), # 16, 5
            nn.Dropout2d(0.25),
            custom2DCNN(64, 128, 2, "same"),
            nn.MaxPool2d((2, 1)), # 8, 5
            nn.Dropout2d(0.25),
            custom2DCNN(128, 128, 2, "same"),
            nn.MaxPool2d(2), # 4, 2
            nn.Dropout2d(0.25),
            custom2DCNN(128, 128, 2, "same"),
            nn.MaxPool2d((2, 1)), #2, 2
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

        lstm_out_last = self.relu(lstm_out_last)

        x1, x2, x3, x4, x5, x6 = torch.split(self.logmel(x), 64, dim=2)

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
    

class v1_mi6_env2_gru(nn.Module):
    def __init__(self, output_nbr, sr):
        super(v1_mi6_env2_gru, self).__init__()

        self.logmel = LogMelSpectrogramLayer(sample_rate=sr, n_mels=420, hop_length=512)
        self.env = EnvelopeFollowingLayerTorchScript(n_fft=2048, hop_length=512, smoothing_factor=4)

        self.relu = nn.ReLU()
        
        self.cnn1 = self._create_cnn_block()
        self.cnn2 = self._create_cnn_block()
        self.cnn3 = self._create_cnn_block()
        self.cnn4 = self._create_cnn_block()
        self.cnn5 = self._create_cnn_block()
        self.cnn6 = self._create_cnn_block()

        self.cnn_env = self._create_cnn_env_block()
        self.gru_env = self._create_gru_env_block()

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
    
    def _create_gru_env_block(self):
        return nn.GRU(input_size=256, hidden_size=112, batch_first=True)

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
        x_env = self.cnn_env(x_env)

        x_env = x_env.permute(0, 2, 1)
        gru_out = self.gru_env(x_env)
        gru_out_last = gru_out[0][:, -1, :].unsqueeze(2).unsqueeze(2)

        gru_out_last = self.relu(gru_out_last)

        x1, x2, x3, x4, x5, x6 = torch.split(self.logmel(x), 70, dim=2)

        x1 = self.cnn1(x1) 
        x2 = self.cnn2(x2)
        x3 = self.cnn3(x3)
        x4 = self.cnn4(x4)
        x5 = self.cnn5(x5)
        x6 = self.cnn6(x6)

        x = torch.cat((x1, x2, x3, x4, x5, x6, gru_out_last), dim=1)

        x_flat = x.view(x.size(0), -1)
        z = self.fc(x_flat)
        return z

class v1_mi6_env2_gru3(nn.Module):
    def __init__(self, output_nbr, sr):
        super(v1_mi6_env2_gru3, self).__init__()

        self.logmel = LogMelSpectrogramLayer(sample_rate=sr, n_mels=384, hop_length=512)
        self.env = EnvelopeFollowingLayerTorchScript(n_fft=2048, hop_length=512, smoothing_factor=4)

        self.relu = nn.ReLU()

        self.cnn1 = self._create_cnn_block()
        self.cnn2 = self._create_cnn_block()
        self.cnn3 = self._create_cnn_block()
        self.cnn4 = self._create_cnn_block()
        self.cnn5 = self._create_cnn_block()
        self.cnn6 = self._create_cnn_block()

        self.cnn_env = self._create_cnn_env_block()
        self.gru_env = self._create_gru_env_block()

        self.fc = nn.Sequential(
            nn.Linear(128 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, 64), 
            nn.ReLU(),
            nn.Linear(64, output_nbr)
        )

    def _create_cnn_env_block(self):
        return nn.Sequential(
            custom1DCNN(1, 32, 8, "same", 4),
            nn.AvgPool1d(8),
            custom1DCNN(32, 64, 6, "same", 3),
            nn.AvgPool1d(7),
            custom1DCNN(64, 128, 4, "same", 2),
            nn.Dropout(0.25),
        )
    
    def _create_gru_env_block(self):
        return nn.GRU(input_size=128, hidden_size=128, batch_first=True)

    def _create_cnn_block(self):
        return nn.Sequential(
            custom2DCNN(1, 32, (2, 3), "same"),
            custom2DCNN(32, 32, (2, 3), "same"),
            nn.MaxPool2d((2, 1)), # 32, 15
            nn.Dropout2d(0.25),
            custom2DCNN(32, 64, (2, 3), "same"),
            custom2DCNN(64, 64, (2, 3), "same"),
            nn.MaxPool2d((2, 3)), # 16, 5
            nn.Dropout2d(0.25),
            custom2DCNN(64, 128, 2, "same"),
            nn.MaxPool2d((2, 1)), # 8, 5
            nn.Dropout2d(0.25),
            custom2DCNN(128, 128, 2, "same"),
            nn.MaxPool2d(2), # 4, 2
            nn.Dropout2d(0.25),
            custom2DCNN(128, 128, 2, "same"),
            nn.MaxPool2d((2, 1)), #2, 2
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
        gru_out = self.gru_env(x_env)
        gru_out_last = gru_out[0][:, -1, :].unsqueeze(2).unsqueeze(2)

        gru_out_last = self.relu(gru_out_last)

        x1, x2, x3, x4, x5, x6 = torch.split(self.logmel(x), 64, dim=2)

        x1 = self.cnn1(x1) 
        x2 = self.cnn2(x2)
        x3 = self.cnn3(x3)
        x4 = self.cnn4(x4)
        x5 = self.cnn5(x5)
        x6 = self.cnn6(x6)

        x = torch.cat((x1, x2, x3, x4, x5, x6, gru_out_last), dim=1)

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