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

from models.layers import LogMelSpectrogramLayer, customCNN2D

class ismir_Ea(nn.Module):
    def __init__(self, output_nbr, args):
        super(ismir_Ea, self).__init__()

        self.sr = args.sr
        self.classnames = args.classnames
        self.fmin = args.fmin
        self.fmax = args.fmax
        self.seglen = args.seglen

        self.logmel = LogMelSpectrogramLayer(sample_rate=self.sr, f_min=self.fmin, f_max=self.fmax, n_mels=420, n_fft=2048, hop_length=128)
        
        self.cnn1 = self._create_cnn_block()

        self.fc = nn.Sequential(
                nn.Linear(160, 80),
                nn.BatchNorm1d(80),
                nn.ReLU(),
                nn.Dropout1d(0.25),
                nn.Linear(80, 40),
                nn.BatchNorm1d(40),
                nn.ReLU(),
                nn.Dropout1d(0.25),
                nn.Linear(40, output_nbr)
            )

    def _create_cnn_block(self):
        return nn.Sequential(
            customCNN2D(1, 40, (4, 8), "same"), 
            customCNN2D(40, 40, (4, 8), "same"),
            nn.MaxPool2d((4, 2)), # 105, 56
            nn.Dropout2d(0.25),
            customCNN2D(40, 80, (3, 7), "same"),
            customCNN2D(80, 80, (3, 7), "same"),
            nn.MaxPool2d((4, 2)), # 26, 28
            nn.Dropout2d(0.25),
            customCNN2D(80, 160, (2, 4), "same"),
            nn.MaxPool2d(2), # 13, 14
            nn.Dropout2d(0.25),
            customCNN2D(160, 160, 2, "same"),
            nn.MaxPool2d(2), # 6, 6
            nn.Dropout2d(0.25),
            customCNN2D(160, 160, (1, 2), "same"),
            nn.MaxPool2d(2), # 3, 3
            nn.Dropout2d(0.25),
            customCNN2D(160, 160, (1, 2), "same"),
            nn.MaxPool2d(3), 
            nn.Dropout2d(0.25),
        )
    
    @torch.jit.export
    def get_attributes(self):
        return self.sr
    
    @torch.jit.export
    def get_classnames(self):
        return self.classnames
    
    @torch.jit.export
    def get_seglen(self):
        return self.seglen

    def forward(self, x):
        x = self.logmel(x)
        x = self.cnn1(x) 
        x_flat = x.view(x.size(0), -1)
        z = self.fc(x_flat)
        return z

class ismir_Eb(nn.Module):
    def __init__(self, output_nbr, args):
        super(ismir_Eb, self).__init__()

        self.sr = args.sr
        self.classnames = args.classnames
        self.fmin = args.fmin
        self.fmax = args.fmax
        self.seglen = args.seglen

        self.logmel = LogMelSpectrogramLayer(sample_rate=self.sr, f_min=self.fmin, f_max=self.fmax, n_mels=420, n_fft=2048, hop_length=128)
        
        self.cnn1 = self._create_cnn_block()

        self.fc = nn.Sequential(
                nn.Linear(160, 80),
                nn.BatchNorm1d(80),
                nn.ReLU(),
                nn.Dropout1d(0.25),
                nn.Linear(80, 40),
                nn.BatchNorm1d(40),
                nn.ReLU(),
                nn.Dropout1d(0.25),
                nn.Linear(40, output_nbr)
            )

    def _create_cnn_block(self):
        return nn.Sequential(
            customCNN2D(1, 40, (4, 8), "same"), 
            customCNN2D(40, 40, (4, 8), "same"),
            nn.MaxPool2d((4, 2)), # 105, 56
            nn.Dropout2d(0.25),
            customCNN2D(40, 80, (3, 7), "same"),
            # customCNN2D(80, 80, (3, 7), "same"),
            nn.MaxPool2d((4, 2)), # 26, 28
            nn.Dropout2d(0.25),
            customCNN2D(80, 160, (2, 4), "same"),
            nn.MaxPool2d(2), # 13, 14
            nn.Dropout2d(0.25),
            customCNN2D(160, 160, 2, "same"),
            nn.MaxPool2d(2), # 6, 6
            nn.Dropout2d(0.25),
            customCNN2D(160, 160, (1, 2), "same"),
            nn.MaxPool2d(2), # 3, 3
            nn.Dropout2d(0.25),
            customCNN2D(160, 160, (1, 2), "same"),
            nn.MaxPool2d(3), 
            nn.Dropout2d(0.25),
        )
    
    @torch.jit.export
    def get_attributes(self):
        return self.sr
    
    @torch.jit.export
    def get_classnames(self):
        return self.classnames
    
    @torch.jit.export
    def get_seglen(self):
        return self.seglen

    def forward(self, x):
        x = self.logmel(x)
        x = self.cnn1(x) 
        x_flat = x.view(x.size(0), -1)
        z = self.fc(x_flat)
        return z


class ismir_Ec(nn.Module):
    def __init__(self, output_nbr, args):
        super(ismir_Ec, self).__init__()

        self.sr = args.sr
        self.classnames = args.classnames
        self.fmin = args.fmin
        self.fmax = args.fmax
        self.seglen = args.seglen

        self.logmel = LogMelSpectrogramLayer(sample_rate=self.sr, f_min=self.fmin, f_max=self.fmax, n_mels=420, n_fft=2048, hop_length=128)
        
        self.cnn1 = self._create_cnn_block()

        self.fc = nn.Sequential(
                nn.Linear(160, 80),
                nn.BatchNorm1d(80),
                nn.ReLU(),
                nn.Dropout1d(0.25),
                nn.Linear(80, 40),
                nn.BatchNorm1d(40),
                nn.ReLU(),
                nn.Dropout1d(0.25),
                nn.Linear(40, output_nbr)
            )

    def _create_cnn_block(self):
        return nn.Sequential(
            customCNN2D(1, 40, (4, 8), "same"), 
            # customCNN2D(40, 40, (4, 8), "same"),
            nn.MaxPool2d((4, 2)), # 105, 56
            nn.Dropout2d(0.25),
            customCNN2D(40, 80, (3, 7), "same"),
            # customCNN2D(80, 80, (3, 7), "same"),
            nn.MaxPool2d((4, 2)), # 26, 28
            nn.Dropout2d(0.25),
            customCNN2D(80, 160, (2, 4), "same"),
            nn.MaxPool2d(2), # 13, 14
            nn.Dropout2d(0.25),
            customCNN2D(160, 160, 2, "same"),
            nn.MaxPool2d(2), # 6, 6
            nn.Dropout2d(0.25),
            customCNN2D(160, 160, (1, 2), "same"),
            nn.MaxPool2d(2), # 3, 3
            nn.Dropout2d(0.25),
            customCNN2D(160, 160, (1, 2), "same"),
            nn.MaxPool2d(3), 
            nn.Dropout2d(0.25),
        )
    
    @torch.jit.export
    def get_attributes(self):
        return self.sr
    
    @torch.jit.export
    def get_classnames(self):
        return self.classnames
    
    @torch.jit.export
    def get_seglen(self):
        return self.seglen

    def forward(self, x):
        x = self.logmel(x)
        x = self.cnn1(x) 
        x_flat = x.view(x.size(0), -1)
        z = self.fc(x_flat)
        return z