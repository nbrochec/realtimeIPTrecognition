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

from models.layers import LogMelSpectrogramLayer, custom2DCNN

class v1(nn.Module):
    def __init__(self, output_nbr):
        super(v1, self).__init__()

        self.logmel = LogMelSpectrogramLayer(sample_rate=24000)
        
        self.cnn = nn.Sequential(
            custom2DCNN(1, 40, (2,3), "same"),
            custom2DCNN(40, 40, (2,3), "same"),
            nn.MaxPool2d((2, 1)),
            custom2DCNN(40, 80, (2,3), "same"),
            custom2DCNN(80, 80, (2,3), "same"),
            nn.MaxPool2d((2, 3)),
            custom2DCNN(80, 160, 2, "same"),
            nn.MaxPool2d((2, 1)),
            custom2DCNN(160, 160, 2, "same"),
            nn.MaxPool2d(2),
            custom2DCNN(160, 160, 2, "same"),
            nn.MaxPool2d((2, 1)),
            custom2DCNN(160, 160, 2, "same"),
            nn.MaxPool2d((4, 2)),
        )

        self.fc = nn.Sequential(
            nn.Linear(160, output_nbr),
        )

    def forward(self, x):
        x = self.logmel(x)
        x = self.cnn(x)
        x_flat = x.view(x.size(0), -1)
        z = self.fc(x_flat)
        return z

class v2(nn.Module):
    def __init__(self, output_nbr):
        super(v2, self).__init__()

        self.logmel = LogMelSpectrogramLayer(sample_rate=24000)
        
        self.cnn = nn.Sequential(
            custom2DCNN(1, 64, (2,3), "same"),
            custom2DCNN(64, 64, (2,3), "same"),
            nn.MaxPool2d((2, 1)),
            custom2DCNN(64, 128, (2,3), "same"),
            custom2DCNN(128, 128, (2,3), "same"),
            nn.MaxPool2d((2, 3)),
            custom2DCNN(128, 256, 2, "same"),
            nn.MaxPool2d((2, 1)),
            custom2DCNN(256, 256, 2, "same"),
            nn.MaxPool2d(2),
            custom2DCNN(256, 512, 2, "same"),
            nn.MaxPool2d((2, 1)),
            custom2DCNN(512, 512, 2, "same"),
            nn.MaxPool2d((2, 1)),
            custom2DCNN(512, 512, 2, "same"),
            nn.MaxPool2d(2),
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

class one_residual(nn.Module):
    def __init__(self, output_nbr):
        super(one_residual, self).__init__()

        self.logmel = LogMelSpectrogramLayer(sample_rate=24000)
        
        self.cnn_part1 = nn.Sequential(
            custom2DCNN(1, 64, (2,3), "same"),
            custom2DCNN(64, 64, (2,3), "same"),
            nn.MaxPool2d((2, 1)),
            custom2DCNN(64, 128, (2,3), "same"),
            custom2DCNN(128, 128, (2,3), "same"),
            nn.MaxPool2d((2, 3)),
            )
        
        self.downsample = nn.MaxPool2d((32, 5))
        
        self.cnn_part2 = nn.Sequential(
            custom2DCNN(128, 256, 2, "same"),
            nn.MaxPool2d((2, 1)),
            custom2DCNN(256, 256, 2, "same"),
            nn.MaxPool2d(2),
            custom2DCNN(256, 512, 2, "same"),
            nn.MaxPool2d((2, 1)),
            custom2DCNN(512, 512, 2, "same"),
            nn.MaxPool2d((2, 1)),
            custom2DCNN(512, 512, 2, "same"),
            nn.MaxPool2d(2),
        )

        self.fc_input_dim = 512 + 128
        self.fc = nn.Sequential(
            nn.Linear(self.fc_input_dim, 512),
            nn.ELU(),
            nn.Linear(512, 256),
            nn.ELU(),
            nn.Linear(256, output_nbr),
        )

    def forward(self, x):
        x = self.logmel(x)
        x = self.cnn_part1(x)
        y = self.downsample(x)
        x = self.cnn_part2(x)
        x_flat = x.view(x.size(0), -1)
        y_flat = y.view(y.size(0), -1)
        z = torch.cat((x_flat, y_flat), dim=1)
        z = self.fc(z)
        return z

class two_residual(nn.Module):
    def __init__(self, output_nbr):
        super(two_residual, self).__init__()

        self.logmel = LogMelSpectrogramLayer(sample_rate=24000)
        
        self.cnn_part1 = nn.Sequential(
            custom2DCNN(1, 64, (2,3), "same"),
            custom2DCNN(64, 64, (2,3), "same"),
            nn.MaxPool2d((2, 1)),
            custom2DCNN(64, 128, (2,3), "same"),
            custom2DCNN(128, 128, (2,3), "same"),
            nn.MaxPool2d((2, 3)),
            )
        
        self.downsample1 = nn.MaxPool2d((32, 5))
        
        self.cnn_part2 = nn.Sequential(
            custom2DCNN(128, 256, 2, "same"),
            nn.MaxPool2d((2, 1)),
            custom2DCNN(256, 256, 2, "same"),
            nn.MaxPool2d(2),
            custom2DCNN(256, 512, 2, "same"),
            nn.MaxPool2d((2, 1)),
        )

        self.downsample2 = nn.MaxPool2d((4, 2))

        self.cnn_part3 = nn.Sequential(
            custom2DCNN(512, 512, 2, "same"),
            nn.MaxPool2d((2, 1)),
            custom2DCNN(512, 512, 2, "same"),
            nn.MaxPool2d(2),
        )

        self.fc_input_dim = (512 * 2) + 128
        self.fc = nn.Sequential(
            nn.Linear(self.fc_input_dim, 512),
            nn.ELU(),
            nn.Linear(512, 256),
            nn.ELU(),
            nn.Linear(256, output_nbr),
        )

    def forward(self, x):
        x = self.logmel(x)
        x = self.cnn_part1(x)
        y = self.downsample1(x)
        x = self.cnn_part2(x)
        z = self.downsample2(x)
        x = self.cnn_part3(x)
        x_flat = x.view(x.size(0), -1)
        y_flat = y.view(y.size(0), -1)
        z_flat = z.view(z.size(0), -1)
        w = torch.cat((x_flat, y_flat, z_flat), dim=1)
        w = self.fc(w)
        return w

class transformer(nn.Module):
    def __init__(self, output_nbr):
        super(transformer, self).__init__()

        self.logmel = LogMelSpectrogramLayer(sample_rate=24000)
        
        self.cnn = nn.Sequential(
            custom2DCNN(1, 64, (2,3), "same"),
            custom2DCNN(64, 64, (2,3), "same"),
            nn.MaxPool2d((2, 1)), 
            custom2DCNN(64, 128, (2,3), "same"),
            custom2DCNN(128, 128, (2,3), "same"),
            nn.MaxPool2d((2, 3)),
            custom2DCNN(128, 256, 2, "same"),
            nn.MaxPool2d((2, 1)), 
            custom2DCNN(256, 256, 2, "same"),
            nn.MaxPool2d(2),
            custom2DCNN(256, 256, 2, "same"),
            nn.MaxPool2d(2),
        )

        self.transformer = nn.Transformer(256, 4, 3, 3)

        self.maxpool2d = nn.MaxPool2d((4, 1)) 

        self.fc = nn.Sequential(
            nn.Linear(256, 128),
            nn.ELU(),
            nn.Linear(128, 64),
            nn.ELU(),
            nn.Linear(64, output_nbr),
        )

    def forward(self, x):
        x = self.logmel(x)
        x = self.cnn(x)
        x = x.permute(2, 0, 1)
        x = self.transformer(x, x)
        x = self.maxpool2d(x)
        x = self.fc(x)
        return x

