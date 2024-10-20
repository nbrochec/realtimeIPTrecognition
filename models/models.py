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

from models.layers import LogMelSpectrogramLayer, customCNN2D, customCNN1D
from utils.constants import SEGMENT_LENGTH

class v1(nn.Module):
    def __init__(self, output_nbr, args):
        super(v1, self).__init__()

        self.sr = args.sr
        self.classnames = args.classnames
        self.fmin = args.fmin
        self.fmax = args.fmax

        self.logmel = LogMelSpectrogramLayer(sample_rate=self.sr, f_min=self.fmin, f_max=self.fmax, n_mels=128, n_fft=2048,  hop_length=512)
        
        self.cnn = nn.Sequential(
            customCNN2D(1, 40, (2,3), "same"),
            customCNN2D(40, 40, (2,3), "same"),
            nn.MaxPool2d((2, 1)),
            nn.Dropout2d(0.25),
            customCNN2D(40, 80, (2,3), "same"),
            customCNN2D(80, 80, (2,3), "same"),
            nn.MaxPool2d((2, 3)),
            nn.Dropout2d(0.25),
            customCNN2D(80, 160, 2, "same"),
            nn.MaxPool2d((2, 1)),
            nn.Dropout2d(0.25),
            customCNN2D(160, 160, 2, "same"),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.25),
            customCNN2D(160, 160, 2, "same"),
            nn.MaxPool2d((2, 1)),
            nn.Dropout2d(0.25),
            customCNN2D(160, 160, 2, "same"),
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
    
    @torch.jit.export
    def get_classnames(self):
        return self.classnames

    def forward(self, x):
        x = self.logmel(x)
        x = self.cnn(x)
        x_flat = x.view(x.size(0), -1)
        z = self.fc(x_flat)
        return z
    
class v1b(nn.Module):
    def __init__(self, output_nbr, args):
        super(v1b, self).__init__()

        self.sr = args.sr
        self.classnames = args.classnames
        self.fmin = args.fmin
        self.fmax = args.fmax

        self.logmel = LogMelSpectrogramLayer(sample_rate=self.sr, f_min=self.fmin, f_max=self.fmax, n_mels=128, n_fft=2048,  hop_length=128)
        
        self.cnn = nn.Sequential(
            customCNN2D(1, 40, (2, 7), "same"),
            customCNN2D(40, 40, (2, 7), "same"),
            nn.MaxPool2d(2), # 64, 57
            nn.Dropout2d(0.25),
            customCNN2D(40, 80, (2, 5), "same"),
            customCNN2D(80, 80, (2, 5), "same"),
            nn.MaxPool2d(2), # 32, 28
            nn.Dropout2d(0.25),
            customCNN2D(80, 160, 2, "same"),
            nn.MaxPool2d(2), # 16, 14
            nn.Dropout2d(0.25),
            customCNN2D(160, 160, 2, "same"),
            nn.MaxPool2d(2), # 8, 7
            nn.Dropout2d(0.25),
            customCNN2D(160, 160, 2, "same"),
            nn.MaxPool2d(2), # 4, 3
            nn.Dropout2d(0.25),
            customCNN2D(160, 160, 2, "same"),
            nn.MaxPool2d((4, 3)),
            nn.Dropout2d(0.25),
        )

        self.fc = nn.Sequential(
            nn.Linear(160, 80),
            nn.BatchNorm1d(80),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(0.25),
            nn.Linear(80, 40),
            nn.BatchNorm1d(40),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(0.25),
            nn.Linear(40, output_nbr)
        )

    @torch.jit.export
    def get_sr(self):
        return self.sr
    
    @torch.jit.export
    def get_classnames(self):
        return self.classnames

    def forward(self, x):
        x = self.logmel(x)
        x = self.cnn(x)
        x_flat = x.view(x.size(0), -1)
        z = self.fc(x_flat)
        return z

class v2(nn.Module):
    def __init__(self, output_nbr, args):
        super(v2, self).__init__()

        self.sr = args.sr
        self.classnames = args.classnames
        self.fmin = args.fmin
        self.fmax = args.fmax

        self.logmel = LogMelSpectrogramLayer(sample_rate=self.sr, f_min=self.fmin, f_max=self.fmax, n_mels=128, n_fft=2048,  hop_length=512)
        
        self.cnn = nn.Sequential(
            customCNN2D(1, 64, (2,3), "same"),
            customCNN2D(64, 64, (2,3), "same"),
            nn.MaxPool2d((2, 1)),
            nn.Dropout2d(0.25),
            customCNN2D(64, 128, (2,3), "same"),
            customCNN2D(128, 128, (2,3), "same"),
            nn.MaxPool2d((2, 3)),
            nn.Dropout2d(0.25),
            customCNN2D(128, 256, 2, "same"),
            nn.MaxPool2d((2, 1)),
            nn.Dropout2d(0.25),
            customCNN2D(256, 256, 2, "same"),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.25),
            customCNN2D(256, 512, 2, "same"),
            nn.MaxPool2d((2, 1)),
            nn.Dropout2d(0.25),
            customCNN2D(512, 512, 2, "same"),
            nn.MaxPool2d((2, 1)),
            nn.Dropout2d(0.25),
            customCNN2D(512, 512, 2, "same"),
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

    @torch.jit.export
    def get_attributes(self):
        return self.sr
    
    @torch.jit.export
    def get_classnames(self):
        return self.classnames

    def forward(self, x):
        x = self.logmel(x)
        x = self.cnn(x)
        x_flat = x.view(x.size(0), -1)
        z = self.fc(x_flat)
        return z
    
class v3(nn.Module):
    def __init__(self, output_nbr, args):
        super(v3, self).__init__()

        self.sr = args.sr
        self.classnames = args.classnames
        self.fmin = args.fmin
        self.fmax = args.fmax

        self.logmel = LogMelSpectrogramLayer(sample_rate=self.sr, f_min=self.fmin, f_max=self.fmax, n_mels=128, n_fft=2048, hop_length=512)
        
        self.cnn = nn.Sequential(
            customCNN2D(1, 64, (2,3), "same"),
            customCNN2D(64, 64, (2,3), "same"),
            nn.MaxPool2d((2, 1)),
            nn.Dropout2d(0.25),
            customCNN2D(64, 128, (2,3), "same"),
            customCNN2D(128, 128, (2,3), "same"),
            nn.MaxPool2d((2, 3)),
            nn.Dropout2d(0.25),
            customCNN2D(128, 256, 2, "same"),
            customCNN2D(256, 256, 2, "same"),
            nn.MaxPool2d((2, 1)),
            nn.Dropout2d(0.25),
            customCNN2D(256, 256, 2, "same"),
            customCNN2D(256, 256, 2, "same"),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.25),
            customCNN2D(256, 512, 2, "same"),
            customCNN2D(512, 512, 2, "same"),
            nn.MaxPool2d((2, 1)),
            nn.Dropout2d(0.25),
            customCNN2D(512, 512, 2, "same"),
            customCNN2D(512, 512, 2, "same"),
            nn.MaxPool2d((2, 1)),
            nn.Dropout2d(0.25),
            customCNN2D(512, 512, 2, "same"),
            customCNN2D(512, 512, 2, "same"),
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

    @torch.jit.export
    def get_attributes(self):
        return self.sr
    
    @torch.jit.export
    def get_classnames(self):
        return self.classnames

    def forward(self, x):
        x = self.logmel(x)
        x = self.cnn(x)
        x_flat = x.view(x.size(0), -1)
        z = self.fc(x_flat)
        return z
 
class v1_mi6_stack2(nn.Module):
    def __init__(self, output_nbr, args):
        super(v1_mi6_stack2, self).__init__()

        self.sr = args.sr
        self.classnames = args.classnames
        self.fmin = args.fmin
        self.fmax = args.fmax

        self.logmel = LogMelSpectrogramLayer(sample_rate=self.sr, f_min=self.fmin, f_max=self.fmax, n_mels=420, n_fft=2048, hop_length=128)
        
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
            customCNN2D(2, 40, (3, 7), "same"), 
            customCNN2D(40, 40, (3, 7), "same"),
            nn.MaxPool2d((2, 1)), # 35, 35
            nn.Dropout2d(0.25),
            customCNN2D(40, 80, (2, 5), "same"),
            customCNN2D(80, 80, (2, 5), "same"),
            nn.MaxPool2d(2), # 17, 17
            nn.Dropout2d(0.25),
            customCNN2D(80, 160, (2, 4), "same"),
            nn.MaxPool2d(2), # 8, 8
            nn.Dropout2d(0.25),
            customCNN2D(160, 160, 2, "same"),
            nn.MaxPool2d(2), # 4, 4
            nn.Dropout2d(0.25),
            customCNN2D(160, 160, (1, 2), "same"),
            nn.MaxPool2d(2), #2, 2
            nn.Dropout2d(0.25),
            customCNN2D(160, 160, (1, 2), "same"),
            nn.MaxPool2d(2), 
            nn.Dropout2d(0.25),
        )
    
    @torch.jit.export
    def get_attributes(self):
        return self.sr
    
    @torch.jit.export
    def get_classnames(self):
        return self.classnames

    def forward(self, x):
        x4_1, x4_2, x4_3, x4_4, x4_5, x4_6 = torch.split(self.logmel(x)[:,:,:, :35], 70, dim=2)
        x5_1, x5_2, x5_3, x5_4, x5_5, x5_6 = torch.split(self.logmel(x)[:,:,:, 22:], 70, dim=2)

        c1 = torch.cat((x4_1, x5_1), dim=1)
        c2 = torch.cat((x4_2, x5_2), dim=1)
        c3 = torch.cat((x4_3, x5_3), dim=1)
        c4 = torch.cat((x4_4, x5_4), dim=1)
        c5 = torch.cat((x4_5, x5_5), dim=1)
        c6 = torch.cat((x4_6, x5_6), dim=1)

        x1 = self.cnn1(c1) 
        x2 = self.cnn2(c2)
        x3 = self.cnn3(c3)
        x4 = self.cnn4(c4)
        x5 = self.cnn5(c5)
        x6 = self.cnn6(c6)

        x = torch.cat((x1, x2, x3, x4, x5, x6), dim=1)

        x_flat = x.view(x.size(0), -1)
        z = self.fc(x_flat)
        return z
