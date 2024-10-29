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
from utils.constants import SEGMENT_LENGTH

class ismir_A(nn.Module):
    def __init__(self, output_nbr, args):
        super(ismir_A, self).__init__()

        self.sr = args.sr
        self.classnames = args.classnames
        self.fmin = args.fmin
        self.fmax = args.fmax

        self.logmel = LogMelSpectrogramLayer(sample_rate=self.sr, f_min=self.fmin, f_max=self.fmax, n_mels=128, n_fft=2048, hop_length=128)
        
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
            customCNN2D(1, 40, (3, 7), "same"), 
            customCNN2D(40, 40, (3, 7), "same"),
            nn.MaxPool2d(2), # 64, 56
            nn.Dropout2d(0.25),
            customCNN2D(40, 80, (2, 5), "same"),
            customCNN2D(80, 80, (2, 5), "same"),
            nn.MaxPool2d(2), # 32, 28
            nn.Dropout2d(0.25),
            customCNN2D(80, 160, (2, 4), "same"),
            nn.MaxPool2d(2), # 16, 14
            nn.Dropout2d(0.25),
            customCNN2D(160, 160, 2, "same"),
            nn.MaxPool2d(2), # 8, 7
            nn.Dropout2d(0.25),
            customCNN2D(160, 160, (1, 2), "same"),
            nn.MaxPool2d(2), # 4, 3
            nn.Dropout2d(0.25),
            customCNN2D(160, 160, (1, 2), "same"),
            nn.MaxPool2d((4, 3)), 
            nn.Dropout2d(0.25),
        )
    
    @torch.jit.export
    def get_attributes(self):
        return self.sr
    
    @torch.jit.export
    def get_classnames(self):
        return self.classnames

    def forward(self, x):
        x = self.logmel(x)
        x = self.cnn1(x) 
        x_flat = x.view(x.size(0), -1)
        z = self.fc(x_flat)
        return z

class ismir_B(nn.Module):
    def __init__(self, output_nbr, args):
        super(ismir_B, self).__init__()

        self.sr = args.sr
        self.classnames = args.classnames
        self.fmin = args.fmin
        self.fmax = args.fmax

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
            customCNN2D(1, 40, (3, 7), "same"), 
            customCNN2D(40, 40, (3, 7), "same"),
            nn.MaxPool2d((4, 2)), # 105, 56
            nn.Dropout2d(0.25),
            customCNN2D(40, 80, (2, 5), "same"),
            customCNN2D(80, 80, (2, 5), "same"),
            nn.MaxPool2d((4, 2)), # 26, 28
            nn.Dropout2d(0.25),
            customCNN2D(80, 160, (2, 4), "same"),
            nn.MaxPool2d(2), # 13, 14
            nn.Dropout2d(0.25),
            customCNN2D(160, 160, 2, "same"),
            nn.MaxPool2d(2), # 7, 6
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

    def forward(self, x):
        x = self.logmel(x)
        x = self.cnn1(x) 
        x_flat = x.view(x.size(0), -1)
        z = self.fc(x_flat)
        return z

class ismir_C(nn.Module):
    def __init__(self, output_nbr, args):
        super(ismir_C, self).__init__()

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
            customCNN2D(1, 40, (3, 7), "same"), 
            customCNN2D(40, 40, (3, 7), "same"),
            nn.MaxPool2d(2), # 35, 113
            nn.Dropout2d(0.25),
            customCNN2D(40, 80, (2, 5), "same"),
            customCNN2D(80, 80, (2, 5), "same"),
            nn.MaxPool2d(2), # 17, 28
            nn.Dropout2d(0.25),
            customCNN2D(80, 160, (2, 4), "same"),
            nn.MaxPool2d(2), # 8, 14
            nn.Dropout2d(0.25),
            customCNN2D(160, 160, 2, "same"),
            nn.MaxPool2d(2), # 4, 7
            nn.Dropout2d(0.25),
            customCNN2D(160, 160, (1, 2), "same"),
            nn.MaxPool2d(2), # 2, 3
            nn.Dropout2d(0.25),
            customCNN2D(160, 160, (1, 2), "same"),
            nn.MaxPool2d((2, 3)), 
            nn.Dropout2d(0.25),
        )
    
    @torch.jit.export
    def get_attributes(self):
        return self.sr
    
    @torch.jit.export
    def get_classnames(self):
        return self.classnames

    def forward(self, x):
        x4_1, x4_2, x4_3, x4_4, x4_5, x4_6 = torch.split(self.logmel(x), 70, dim=2)

        x1 = self.cnn1(x4_1) 
        x2 = self.cnn2(x4_2)
        x3 = self.cnn3(x4_3)
        x4 = self.cnn4(x4_4)
        x5 = self.cnn5(x4_5)
        x6 = self.cnn6(x4_6)

        x = torch.cat((x1, x2, x3, x4, x5, x6), dim=1)

        x_flat = x.view(x.size(0), -1)
        z = self.fc(x_flat)
        return z

class ismir_D(nn.Module):
    def __init__(self, output_nbr, args):
        super(ismir_D, self).__init__()

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
            nn.MaxPool2d(2), # 35, 35
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
        x4_1, x4_2, x4_3, x4_4, x4_5, x4_6 = torch.split(self.logmel(x)[:,:,:, :70], 70, dim=2)
        x5_1, x5_2, x5_3, x5_4, x5_5, x5_6 = torch.split(self.logmel(x)[:,:,:, 43:], 70, dim=2)

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


class ismir_E(nn.Module):
    def __init__(self, output_nbr, args):
        super(ismir_E, self).__init__()

        self.sr = args.sr
        self.classnames = args.classnames
        self.fmin = args.fmin
        self.fmax = args.fmax

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
            nn.MaxPool2d(2), # 7, 6
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

    def forward(self, x):
        x = self.logmel(x)
        x = self.cnn1(x) 
        x_flat = x.view(x.size(0), -1)
        z = self.fc(x_flat)
        return z
    
class ismir_F(nn.Module):
    def __init__(self, output_nbr, args):
        super(ismir_F, self).__init__()

        self.sr = args.sr
        self.classnames = args.classnames
        self.fmin = args.fmin
        self.fmax = args.fmax

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
            customCNN2D(2, 40, (3, 7), "same"), 
            customCNN2D(40, 40, (3, 7), "same"),
            nn.MaxPool2d((4, 2)), # 105, 35
            nn.Dropout2d(0.25),
            customCNN2D(40, 80, (2, 5), "same"),
            customCNN2D(80, 80, (2, 5), "same"),
            nn.MaxPool2d((4, 2)), # 26, 17
            nn.Dropout2d(0.25),
            customCNN2D(80, 160, (2, 4), "same"),
            nn.MaxPool2d(2), # 13, 8
            nn.Dropout2d(0.25),
            customCNN2D(160, 160, 2, "same"),
            nn.MaxPool2d(2), # 7, 4
            nn.Dropout2d(0.25),
            customCNN2D(160, 160, (1, 2), "same"),
            nn.MaxPool2d(2), # 3, 2
            nn.Dropout2d(0.25),
            customCNN2D(160, 160, (1, 2), "same"),
            nn.MaxPool2d((3, 2)), 
            nn.Dropout2d(0.25),
        )
    
    @torch.jit.export
    def get_attributes(self):
        return self.sr
    
    @torch.jit.export
    def get_classnames(self):
        return self.classnames

    def forward(self, x):
        x1 = self.logmel(x)[:,:,:,:70]
        x2 = self.logmel(x)[:,:,:,43:]

        x = torch.cat((x1,x2), dim=1)
        x = self.cnn1(x) 
        x_flat = x.view(x.size(0), -1)
        z = self.fc(x_flat)
        return z
    

class ismir_G(nn.Module):
    def __init__(self, output_nbr, args):
        super(ismir_G, self).__init__()

        self.sr = args.sr
        self.classnames = args.classnames
        self.fmin = args.fmin
        self.fmax = args.fmax

        self.logmel = LogMelSpectrogramLayer(sample_rate=self.sr//2, f_min=self.fmin, f_max=self.fmax, n_mels=420, n_fft=2048, hop_length=128)
        
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
            customCNN2D(1, 40, (4, 10), "same"), 
            customCNN2D(40, 40, (4, 10), "same"),
            nn.MaxPool2d((4, 2)), # 105, 56
            nn.Dropout2d(0.25),
            customCNN2D(40, 80, (3, 7), "same"),
            customCNN2D(80, 80, (3, 7), "same"),
            nn.MaxPool2d((4, 2)), # 26, 28
            nn.Dropout2d(0.25),
            customCNN2D(80, 160, (2, 4), "same"),
            nn.MaxPool2d(2), # 13, 14
            nn.Dropout2d(0.25),
            customCNN2D(160, 160, (2, 3), "same"),
            nn.MaxPool2d(2), # 7, 6
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

    def forward(self, x):
        x = self.logmel(x)
        x = self.cnn1(x) 
        x_flat = x.view(x.size(0), -1)
        z = self.fc(x_flat)
        return z