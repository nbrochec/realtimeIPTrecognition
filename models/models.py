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

from models.layers import LogMelSpectrogramLayer, custom2DCNN, custom1DCNN, EnvelopeFollowingLayerTorchScript, customARB, LogMelSpectrogramLayerERANN, customARB1D
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
    

class v1_mi6_env2_mod_new_stack(nn.Module):
    def __init__(self, output_nbr, args):
        super(v1_mi6_env2_mod_new_stack, self).__init__()

        self.sr = args.sr

        # self.logmel1 = LogMelSpectrogramLayer(sample_rate=self.sr, n_mels=420, n_fft=2048, hop_length=512)
        # self.logmel2 = LogMelSpectrogramLayer(sample_rate=self.sr, n_mels=420, n_fft=2048, hop_length=256)
        self.logmel3 = LogMelSpectrogramLayer(sample_rate=self.sr, n_mels=420, n_fft=2048, hop_length=128)
        self.env = EnvelopeFollowingLayerTorchScript(n_fft=2048, hop_length=128, smoothing_factor=4)
        
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
        # augment kernel
        return nn.Sequential(
            custom1DCNN(1, 40, 20, "same", 1), 
            nn.AvgPool1d(8),
            nn.Dropout1d(0.25),
            custom1DCNN(40, 40, 15, "same", 2), 
            nn.AvgPool1d(8),
            nn.Dropout1d(0.25),
            custom1DCNN(40, 80, 10, "same", 2),  #new
            nn.AvgPool1d(8), #new
            nn.Dropout1d(0.25), #new
            custom1DCNN(80, 80, 7, "same", 3), 
            nn.AvgPool1d(2),
            nn.Dropout1d(0.25),
            custom1DCNN(80, 160, 2, "same", 4),
            nn.AvgPool1d(7),
            nn.Dropout1d(0.25),
        )

    def _create_cnn_block(self):
        return nn.Sequential(
            custom2DCNN(4, 40, (3, 7), "same"), 
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
            custom2DCNN(160, 160, 2, "same"),
            nn.MaxPool2d(2), # 4, 2
            nn.Dropout2d(0.25),
            custom2DCNN(160, 160, (1, 2), "same"),
            nn.MaxPool2d((4, 2)), #2, 2 #old 2,1
            nn.Dropout2d(0.25),
            # custom2DCNN(160, 160, (1, 2), "same"), # old 2
            # nn.MaxPool2d(2),
            # nn.Dropout2d(0.25),
        )

    def forward(self, x):
        x_env = self.env(x)
        x_env = x_env[:, :, :-1]
        x_env = self.cnn_env(x_env)

        x4_1, x4_2, x4_3, x4_4, x4_5, x4_6 = torch.split(self.logmel3(x)[:,:,:, :15], 70, dim=2)
        x5_1, x5_2, x5_3, x5_4, x5_5, x5_6 = torch.split(self.logmel3(x)[:,:,:, 14:29], 70, dim=2)
        x6_1, x6_2, x6_3, x6_4, x6_5, x6_6 = torch.split(self.logmel3(x)[:,:,:, 28:43], 70, dim=2)
        x7_1, x7_2, x7_3, x7_4, x7_5, x7_6 = torch.split(self.logmel3(x)[:,:,:, 42:], 70, dim=2)

        c1 = torch.cat((x4_1, x5_1, x6_1, x7_1), dim=1)
        c2 = torch.cat((x4_2, x5_2, x6_2, x7_2), dim=1)
        c3 = torch.cat((x4_3, x5_3, x6_3, x7_3), dim=1)
        c4 = torch.cat((x4_4, x5_4, x6_4, x7_4), dim=1)
        c5 = torch.cat((x4_5, x5_5, x6_5, x7_5), dim=1)
        c6 = torch.cat((x4_6, x5_6, x6_6, x7_6), dim=1)

        x1 = self.cnn1(c1) 
        x2 = self.cnn2(c2)
        x3 = self.cnn3(c3)
        x4 = self.cnn4(c4)
        x5 = self.cnn5(c5)
        x6 = self.cnn6(c6)

        x = torch.cat((x1, x2, x3, x4, x5, x6, x_env.unsqueeze(3)), dim=1)

        x_flat = x.view(x.size(0), -1)
        z = self.fc(x_flat)
        return z
    
class v1_mi6_env2_mod_new_stack8(nn.Module):
    def __init__(self, output_nbr, args):
        super(v1_mi6_env2_mod_new_stack8, self).__init__()

        self.sr = args.sr

        # self.logmel1 = LogMelSpectrogramLayer(sample_rate=self.sr, n_mels=420, n_fft=2048, hop_length=512)
        # self.logmel2 = LogMelSpectrogramLayer(sample_rate=self.sr, n_mels=420, n_fft=2048, hop_length=256)
        self.logmel3 = LogMelSpectrogramLayer(sample_rate=self.sr, n_mels=420, n_fft=2048, hop_length=64)
        self.env = EnvelopeFollowingLayerTorchScript(n_fft=2048, hop_length=128, smoothing_factor=4)
        
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
                nn.LeakyReLU(negative_slope=0.01),
                nn.Dropout1d(0.25),
                nn.Linear(320, 80),
                nn.BatchNorm1d(80),
                nn.LeakyReLU(negative_slope=0.01),
                nn.Dropout1d(0.25),
                nn.Linear(80, output_nbr)
            )

    def _create_cnn_env_block(self):
        # change dilation
        # augment kernel
        return nn.Sequential(
            custom1DCNN(1, 40, 20, "same", 1), 
            nn.AvgPool1d(16),
            nn.Dropout1d(0.25),
            custom1DCNN(40, 40, 15, "same", 2), 
            nn.AvgPool1d(8),
            nn.Dropout1d(0.25),
            custom1DCNN(40, 80, 7, "same", 3), 
            nn.AvgPool1d(8),
            nn.Dropout1d(0.25),
            custom1DCNN(80, 160, 2, "same", 4),
            nn.AvgPool1d(7),
            nn.Dropout1d(0.25),
        )

    def _create_cnn_block(self):
        return nn.Sequential(
            custom2DCNN(8, 40, (3, 7), "same"), 
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
            custom2DCNN(160, 160, 2, "same"),
            nn.MaxPool2d(2), # 4, 2
            nn.Dropout2d(0.25),
            custom2DCNN(160, 160, (1, 2), "same"),
            nn.MaxPool2d((4, 2)), #2, 2 #old 2,1
            nn.Dropout2d(0.25),
            # custom2DCNN(160, 160, (1, 2), "same"), # old 2
            # nn.MaxPool2d(2),
            # nn.Dropout2d(0.25),
        )

    def forward(self, x):
        x_env = self.env(x)
        x_env = x_env[:, :, :-1]
        x_env = self.cnn_env(x_env)

        x4_1, x4_2, x4_3, x4_4, x4_5, x4_6 = torch.split(self.logmel3(x)[:,:,:, :15], 70, dim=2)
        x5_1, x5_2, x5_3, x5_4, x5_5, x5_6 = torch.split(self.logmel3(x)[:,:,:, 14:29], 70, dim=2)
        x6_1, x6_2, x6_3, x6_4, x6_5, x6_6 = torch.split(self.logmel3(x)[:,:,:, 28:43], 70, dim=2)
        x7_1, x7_2, x7_3, x7_4, x7_5, x7_6 = torch.split(self.logmel3(x)[:,:,:, 42:57], 70, dim=2)
        x8_1, x8_2, x8_3, x8_4, x8_5, x8_6 = torch.split(self.logmel3(x)[:,:,:, 56:71], 70, dim=2)
        x9_1, x9_2, x9_3, x9_4, x9_5, x9_6 = torch.split(self.logmel3(x)[:,:,:, 70:85], 70, dim=2)
        x10_1, x10_2, x10_3, x10_4, x10_5, x10_6 = torch.split(self.logmel3(x)[:,:,:, 84:99], 70, dim=2)
        x11_1, x11_2, x11_3, x11_4, x11_5, x11_6 = torch.split(self.logmel3(x)[:,:,:, 98:113], 70, dim=2)

        c1 = torch.cat((x4_1, x5_1, x6_1, x7_1, x8_1, x9_1, x10_1, x11_1), dim=1)
        c2 = torch.cat((x4_2, x5_2, x6_2, x7_2, x8_2, x9_2, x10_2, x11_2), dim=1)
        c3 = torch.cat((x4_3, x5_3, x6_3, x7_3, x8_3, x9_3, x10_3, x11_3), dim=1)
        c4 = torch.cat((x4_4, x5_4, x6_4, x7_4, x8_4, x9_4, x10_4, x11_4), dim=1)
        c5 = torch.cat((x4_5, x5_5, x6_5, x7_5, x8_5, x9_5, x10_5, x11_5), dim=1)
        c6 = torch.cat((x4_6, x5_6, x6_6, x7_6, x8_6, x9_6, x10_6, x11_6), dim=1)

        x1 = self.cnn1(c1) 
        x2 = self.cnn2(c2)
        x3 = self.cnn3(c3)
        x4 = self.cnn4(c4)
        x5 = self.cnn5(c5)
        x6 = self.cnn6(c6)

        x = torch.cat((x1, x2, x3, x4, x5, x6, x_env.unsqueeze(3)), dim=1)

        x_flat = x.view(x.size(0), -1)
        z = self.fc(x_flat)
        return z
    
class v1_mi6_env2_mod_new_stack8x8(nn.Module):
    def __init__(self, output_nbr, args):
        super(v1_mi6_env2_mod_new_stack8x8, self).__init__()

        self.sr = args.sr

        self.logmel3 = LogMelSpectrogramLayer(sample_rate=self.sr, n_mels=400, n_fft=2048, hop_length=64)
        self.env = EnvelopeFollowingLayerTorchScript(n_fft=2048, hop_length=64, smoothing_factor=4)
        
        self.cnn1 = self._create_cnn_block()
        self.cnn2 = self._create_cnn_block()
        self.cnn3 = self._create_cnn_block()
        self.cnn4 = self._create_cnn_block()
        self.cnn5 = self._create_cnn_block()
        self.cnn6 = self._create_cnn_block()
        self.cnn7 = self._create_cnn_block()
        self.cnn8 = self._create_cnn_block()

        self.cnn_env = self._create_cnn_env_block()

        self.fc = nn.Sequential(
                nn.Linear(160 * 9, 480),
                nn.BatchNorm1d(480),
                nn.LeakyReLU(negative_slope=0.01),
                nn.Dropout1d(0.25),
                nn.Linear(480, 160),
                nn.BatchNorm1d(160),
                nn.LeakyReLU(negative_slope=0.01),
                nn.Dropout1d(0.25),
                nn.Linear(160, output_nbr)
            )

    def _create_cnn_env_block(self):
        # change dilation
        # augment kernel
        return nn.Sequential(
            custom1DCNN(1, 40, 20, "same", 1), 
            nn.AvgPool1d(16),
            nn.Dropout1d(0.25),
            custom1DCNN(40, 40, 15, "same", 2), 
            nn.AvgPool1d(8),
            nn.Dropout1d(0.25),
            custom1DCNN(40, 80, 7, "same", 3), 
            nn.AvgPool1d(8),
            nn.Dropout1d(0.25),
            custom1DCNN(80, 160, 2, "same", 4),
            nn.AvgPool1d(7),
            nn.Dropout1d(0.25),
        )

    def _create_cnn_block(self):
        return nn.Sequential(
            custom2DCNN(8, 40, (3, 7), "same"), 
            custom2DCNN(40, 40, (3, 7), "same"),
            nn.MaxPool2d((2, 1)), # 25, 15
            nn.Dropout2d(0.25),
            custom2DCNN(40, 80, (2, 5), "same"),
            custom2DCNN(80, 80, (2, 5), "same"),
            nn.MaxPool2d((2, 3)), # 12, 5
            nn.Dropout2d(0.25),
            custom2DCNN(80, 160, (2, 3), "same"),
            nn.MaxPool2d((2, 1)), # 6, 5
            nn.Dropout2d(0.25),
            custom2DCNN(160, 160, 2, "same"),
            nn.MaxPool2d(2), # 3, 2
            nn.Dropout2d(0.25),
            custom2DCNN(160, 160, (1, 2), "same"),
            nn.MaxPool2d((3, 2)), #2, 2 #old 2,1
            nn.Dropout2d(0.25),
            # custom2DCNN(160, 160, (1, 2), "same"), # old 2
            # nn.MaxPool2d(2),
            # nn.Dropout2d(0.25),
        )

    def forward(self, x):
        x_env = self.env(x)
        x_env = x_env[:, :, :-1]
        x_env = self.cnn_env(x_env)

        x4_1, x4_2, x4_3, x4_4, x4_5, x4_6, x4_7, x4_8 = torch.split(self.logmel3(x)[:,:,:, :15], 50, dim=2)
        x5_1, x5_2, x5_3, x5_4, x5_5, x5_6, x5_7, x5_8 = torch.split(self.logmel3(x)[:,:,:, 14:29], 50, dim=2)
        x6_1, x6_2, x6_3, x6_4, x6_5, x6_6, x6_7, x6_8 = torch.split(self.logmel3(x)[:,:,:, 28:43], 50, dim=2)
        x7_1, x7_2, x7_3, x7_4, x7_5, x7_6, x7_7, x7_8 = torch.split(self.logmel3(x)[:,:,:, 42:57], 50, dim=2)
        x8_1, x8_2, x8_3, x8_4, x8_5, x8_6, x8_7, x8_8 = torch.split(self.logmel3(x)[:,:,:, 56:71], 50, dim=2)
        x9_1, x9_2, x9_3, x9_4, x9_5, x9_6, x9_7, x9_8 = torch.split(self.logmel3(x)[:,:,:, 70:85], 50, dim=2)
        x10_1, x10_2, x10_3, x10_4, x10_5, x10_6, x10_7, x10_8 = torch.split(self.logmel3(x)[:,:,:, 84:99], 50, dim=2)
        x11_1, x11_2, x11_3, x11_4, x11_5, x11_6, x11_7, x11_8 = torch.split(self.logmel3(x)[:,:,:, 98:113], 50, dim=2)

        c1 = torch.cat((x4_1, x5_1, x6_1, x7_1, x8_1, x9_1, x10_1, x11_1), dim=1)
        c2 = torch.cat((x4_2, x5_2, x6_2, x7_2, x8_2, x9_2, x10_2, x11_2), dim=1)
        c3 = torch.cat((x4_3, x5_3, x6_3, x7_3, x8_3, x9_3, x10_3, x11_3), dim=1)
        c4 = torch.cat((x4_4, x5_4, x6_4, x7_4, x8_4, x9_4, x10_4, x11_4), dim=1)
        c5 = torch.cat((x4_5, x5_5, x6_5, x7_5, x8_5, x9_5, x10_5, x11_5), dim=1)
        c6 = torch.cat((x4_6, x5_6, x6_6, x7_6, x8_6, x9_6, x10_6, x11_6), dim=1)
        c7 = torch.cat((x4_7, x5_7, x6_7, x7_7, x8_7, x9_7, x10_7, x11_7), dim=1)
        c8 = torch.cat((x4_8, x5_8, x6_8, x7_8, x8_8, x9_8, x10_8, x11_8), dim=1)

        x1 = self.cnn1(c1) 
        x2 = self.cnn2(c2)
        x3 = self.cnn3(c3)
        x4 = self.cnn4(c4)
        x5 = self.cnn5(c5)
        x6 = self.cnn6(c6)
        x7 = self.cnn7(c7)
        x8 = self.cnn8(c8)

        x = torch.cat((x1, x2, x3, x4, x5, x6, x7, x8, x_env.unsqueeze(3)), dim=1)

        x_flat = x.view(x.size(0), -1)
        z = self.fc(x_flat)
        return z
    
class ARNModel_new(nn.Module):
    def __init__(self, output_nbr, args):
        super(ARNModel_new, self).__init__()
        
        self.sr = args.sr
        self.logmel = LogMelSpectrogramLayer(n_fft=2048, hop_length=64, sample_rate=self.sr, n_mels=420)
        self.output_nbr = output_nbr

        self.env1 = EnvelopeFollowingLayerTorchScript(n_fft=2048, hop_length=64, smoothing_factor=4)

        self.arb1 = self._create_ARB_net()
        self.arb2 = self._create_ARB_net()
        self.arb3 = self._create_ARB_net()
        self.arb4 = self._create_ARB_net()
        self.arb5 = self._create_ARB_net()
        self.arb6 = self._create_ARB_net()

        self.fc = self._create_fc_block()

        self.cnn_env = self._create_env_block()

    def _create_fc_block(self):
        return nn.Sequential(
            nn.Linear(160 * 7, 320),
            nn.BatchNorm1d(320),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(320, 80),
            nn.BatchNorm1d(80),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(80, self.output_nbr)
        )

    def _create_ARB_net(self):
        return nn.Sequential(
            customARB(8, 40, (3, 7), first=True),
            customARB(40, 40, (3, 7), (2, 1)),

            customARB(40, 80, (2, 5)),
            customARB(80, 80, (2, 5), (2, 3)),

            customARB(80, 160, (2, 3), (2, 1)),
            customARB(160, 160, 2, 2),
            customARB(160, 160, (1, 2), (4, 2))

        )
    
    def _create_env_block(self):
        return nn.Sequential(
            customARB1D(1, 40, 20, 1, 16, first=True),
            customARB1D(40, 40, 15, 2, 8),
            customARB1D(40, 80, 7, 3, 8),
            customARB1D(80, 160, 2, 4, 7)
        )

    def forward(self, x):
        x_env = self.env1(x)[:, :, :-1]
        x_env = self.cnn_env(x_env)

        x4_1, x4_2, x4_3, x4_4, x4_5, x4_6 = torch.split(self.logmel(x)[:,:,:, :15], 70, dim=2)
        x5_1, x5_2, x5_3, x5_4, x5_5, x5_6 = torch.split(self.logmel(x)[:,:,:, 14:29], 70, dim=2)
        x6_1, x6_2, x6_3, x6_4, x6_5, x6_6 = torch.split(self.logmel(x)[:,:,:, 28:43], 70, dim=2)
        x7_1, x7_2, x7_3, x7_4, x7_5, x7_6 = torch.split(self.logmel(x)[:,:,:, 42:57], 70, dim=2)
        x8_1, x8_2, x8_3, x8_4, x8_5, x8_6 = torch.split(self.logmel(x)[:,:,:, 56:71], 70, dim=2)
        x9_1, x9_2, x9_3, x9_4, x9_5, x9_6 = torch.split(self.logmel(x)[:,:,:, 70:85], 70, dim=2)
        x10_1, x10_2, x10_3, x10_4, x10_5, x10_6 = torch.split(self.logmel(x)[:,:,:, 84:99], 70, dim=2)
        x11_1, x11_2, x11_3, x11_4, x11_5, x11_6 = torch.split(self.logmel(x)[:,:,:, 98:113], 70, dim=2)

        c1 = torch.cat((x4_1, x5_1, x6_1, x7_1, x8_1, x9_1, x10_1, x11_1), dim=1)
        c2 = torch.cat((x4_2, x5_2, x6_2, x7_2, x8_2, x9_2, x10_2, x11_2), dim=1)
        c3 = torch.cat((x4_3, x5_3, x6_3, x7_3, x8_3, x9_3, x10_3, x11_3), dim=1)
        c4 = torch.cat((x4_4, x5_4, x6_4, x7_4, x8_4, x9_4, x10_4, x11_4), dim=1)
        c5 = torch.cat((x4_5, x5_5, x6_5, x7_5, x8_5, x9_5, x10_5, x11_5), dim=1)
        c6 = torch.cat((x4_6, x5_6, x6_6, x7_6, x8_6, x9_6, x10_6, x11_6), dim=1)

        x1 = self.arb1(c1) 
        x2 = self.arb2(c2)
        x3 = self.arb3(c3)
        x4 = self.arb4(c4)
        x5 = self.arb5(c5)
        x6 = self.arb6(c6)

        x_cat = torch.cat((x1, x2, x3, x4, x5, x6, x_env.unsqueeze(2)), dim=1)
        x = torch.flatten(x_cat, 1) 
        x = self.fc(x)
        return x
    

class ARNModel_mod_new(nn.Module):
    def __init__(self, output_nbr, args):
        super(ARNModel_mod_new, self).__init__()
        
        self.sr = args.sr
        self.logmel = LogMelSpectrogramLayer(n_fft=2048, hop_length=128, sample_rate=self.sr, n_mels=420)
        self.output_nbr = output_nbr

        self.env1 = EnvelopeFollowingLayerTorchScript(n_fft=2048, hop_length=128, smoothing_factor=4)

        self.arb1 = self._create_ARB_net()
        self.arb2 = self._create_ARB_net()
        self.arb3 = self._create_ARB_net()
        self.arb4 = self._create_ARB_net()
        self.arb5 = self._create_ARB_net()
        self.arb6 = self._create_ARB_net()

        self.fc = self._create_fc_block()

        self.cnn_env = self._create_env_block()

    def _create_fc_block(self):
        return nn.Sequential(
            nn.Linear(160 * 7, 320),
            nn.BatchNorm1d(320),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(320, 80),
            nn.BatchNorm1d(80),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(80, self.output_nbr)
        )

    def _create_ARB_net(self):
        return nn.Sequential(
            customARB(8, 40, (2, 7), first=True), # 2, 7
            customARB(40, 80, (2, 5), (2, 1)),
            customARB(80, 160, (2, 3), (2, 3)),
            customARB(160, 160, (1, 2), (2, 1)),
            customARB(160, 160, (1, 2), 2),
            customARB(160, 160, (1, 2), (4, 2))
        )
    
    def _create_env_block(self):
        return nn.Sequential(
            customARB1D(1, 40, 20, 1, 16, first=True),
            customARB1D(40, 40, 15, 2, 8),
            customARB1D(40, 80, 7, 3, 8),
            customARB1D(80, 160, 2, 4, 7)
        )

    def forward(self, x):
        x_env = self.env1(x)[:, :, :-1]
        x_env = self.cnn_env(x_env)

        x4_1, x4_2, x4_3, x4_4, x4_5, x4_6 = torch.split(self.logmel(x)[:,:,:, :15], 70, dim=2)
        x5_1, x5_2, x5_3, x5_4, x5_5, x5_6 = torch.split(self.logmel(x)[:,:,:, 14:29], 70, dim=2)
        x6_1, x6_2, x6_3, x6_4, x6_5, x6_6 = torch.split(self.logmel(x)[:,:,:, 28:43], 70, dim=2)
        x7_1, x7_2, x7_3, x7_4, x7_5, x7_6 = torch.split(self.logmel(x)[:,:,:, 42:], 70, dim=2)

        c1 = torch.cat((x4_1, x5_1, x6_1, x7_1), dim=1)
        c2 = torch.cat((x4_2, x5_2, x6_2, x7_2), dim=1)
        c3 = torch.cat((x4_3, x5_3, x6_3, x7_3), dim=1)
        c4 = torch.cat((x4_4, x5_4, x6_4, x7_4), dim=1)
        c5 = torch.cat((x4_5, x5_5, x6_5, x7_5), dim=1)
        c6 = torch.cat((x4_6, x5_6, x6_6, x7_6), dim=1)

        x1 = self.arb1(c1) 
        x2 = self.arb2(c2)
        x3 = self.arb3(c3)
        x4 = self.arb4(c4)
        x5 = self.arb5(c5)
        x6 = self.arb6(c6)

        x_cat = torch.cat((x1, x2, x3, x4, x5, x6, x_env.unsqueeze(2)), dim=1)
        x = torch.flatten(x_cat, 1) 
        x = self.fc(x)
        return x
    

class ARNModel_mix(nn.Module):
    def __init__(self, output_nbr, args):
        super(ARNModel_mix, self).__init__()
        
        self.sr = args.sr
        self.logmel = LogMelSpectrogramLayer(n_fft=2048, hop_length=64, sample_rate=self.sr, n_mels=420)
        self.output_nbr = output_nbr

        self.env1 = EnvelopeFollowingLayerTorchScript(n_fft=2048, hop_length=64, smoothing_factor=4)

        self.arb1 = self._create_ARB_net()
        self.arb2 = self._create_ARB_net()
        self.arb3 = self._create_ARB_net()
        self.arb4 = self._create_ARB_net()
        self.arb5 = self._create_ARB_net()
        self.arb6 = self._create_ARB_net()

        self.fc = self._create_fc_block()

        self.cnn_env = self._create_env_block()

    def _create_fc_block(self):
        return nn.Sequential(
            nn.Linear(160 * 7, 320),
            nn.BatchNorm1d(320),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(320, 80),
            nn.BatchNorm1d(80),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(80, self.output_nbr)
        )

    def _create_ARB_net(self):
        return nn.Sequential(
            custom2DCNN(8, 40, (3, 7), "same"), 
            custom2DCNN(40, 40, (3, 7), "same"),
            nn.MaxPool2d((2, 1)), # 25, 15
            nn.Dropout2d(0.25),
            custom2DCNN(40, 80, (2, 5), "same"),
            custom2DCNN(80, 80, (2, 5), "same"),
            nn.MaxPool2d((2, 3)), # 12, 5
            nn.Dropout2d(0.25),
            custom2DCNN(80, 160, (2, 3), "same"),
            nn.MaxPool2d((2, 1)), # 6, 5
            nn.Dropout2d(0.25),
            custom2DCNN(160, 160, 2, "same"),
            nn.MaxPool2d(2), # 3, 2
            nn.Dropout2d(0.25),
            custom2DCNN(160, 160, (1, 2), "same"),
            nn.MaxPool2d((3, 2)), #2, 2 #old 2,1
            nn.Dropout2d(0.25),
        )
    
    def _create_env_block(self):
        return nn.Sequential(
            customARB1D(1, 40, 20, 1, 16, first=True),
            customARB1D(40, 40, 15, 2, 8),
            customARB1D(40, 80, 7, 3, 8),
            customARB1D(80, 160, 2, 4, 7)
        )

    def forward(self, x):
        x_env = self.env1(x)[:, :, :-1]
        x_env = self.cnn_env(x_env)

        x4_1, x4_2, x4_3, x4_4, x4_5, x4_6 = torch.split(self.logmel(x)[:,:,:, :15], 70, dim=2)
        x5_1, x5_2, x5_3, x5_4, x5_5, x5_6 = torch.split(self.logmel(x)[:,:,:, 14:29], 70, dim=2)
        x6_1, x6_2, x6_3, x6_4, x6_5, x6_6 = torch.split(self.logmel(x)[:,:,:, 28:43], 70, dim=2)
        x7_1, x7_2, x7_3, x7_4, x7_5, x7_6 = torch.split(self.logmel(x)[:,:,:, 42:57], 70, dim=2)
        x8_1, x8_2, x8_3, x8_4, x8_5, x8_6 = torch.split(self.logmel(x)[:,:,:, 56:71], 70, dim=2)
        x9_1, x9_2, x9_3, x9_4, x9_5, x9_6 = torch.split(self.logmel(x)[:,:,:, 70:85], 70, dim=2)
        x10_1, x10_2, x10_3, x10_4, x10_5, x10_6 = torch.split(self.logmel(x)[:,:,:, 84:99], 70, dim=2)
        x11_1, x11_2, x11_3, x11_4, x11_5, x11_6 = torch.split(self.logmel(x)[:,:,:, 98:113], 70, dim=2)

        c1 = torch.cat((x4_1, x5_1, x6_1, x7_1, x8_1, x9_1, x10_1, x11_1), dim=1)
        c2 = torch.cat((x4_2, x5_2, x6_2, x7_2, x8_2, x9_2, x10_2, x11_2), dim=1)
        c3 = torch.cat((x4_3, x5_3, x6_3, x7_3, x8_3, x9_3, x10_3, x11_3), dim=1)
        c4 = torch.cat((x4_4, x5_4, x6_4, x7_4, x8_4, x9_4, x10_4, x11_4), dim=1)
        c5 = torch.cat((x4_5, x5_5, x6_5, x7_5, x8_5, x9_5, x10_5, x11_5), dim=1)
        c6 = torch.cat((x4_6, x5_6, x6_6, x7_6, x8_6, x9_6, x10_6, x11_6), dim=1)

        x1 = self.arb1(c1) 
        x2 = self.arb2(c2)
        x3 = self.arb3(c3)
        x4 = self.arb4(c4)
        x5 = self.arb5(c5)
        x6 = self.arb6(c6)

        x_cat = torch.cat((x1, x2, x3, x4, x5, x6, x_env.unsqueeze(2)), dim=1)
        x = torch.flatten(x_cat, 1) 
        x = self.fc(x)
        return x
    
