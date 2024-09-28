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

from models.layers import LogMelSpectrogramLayer, custom2DCNN, custom1DCNN, EnvelopeExtractor, spectralEnergyExtractor, EnvelopeFollowingLayerTorchScript
from models.layers import EnvelopeMelLayer
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
    
class v2bis(nn.Module):
    def __init__(self, output_nbr, sr):
        super(v2bis, self).__init__()

        self.logmel = LogMelSpectrogramLayer(sample_rate=sr)
        
        self.cnn = nn.Sequential(
            custom2DCNN(1, 64, (2,3), "same"),
            custom2DCNN(64, 64, (2,3), "same"),
            nn.AvgPool2d((2, 1)),
            nn.Dropout2d(0.25),
            custom2DCNN(64, 128, (2,3), "same"),
            custom2DCNN(128, 128, (2,3), "same"),
            nn.AvgPool2d((2, 3)),
            nn.Dropout2d(0.25),
            custom2DCNN(128, 256, 2, "same"),
            nn.AvgPool2d((2, 1)),
            nn.Dropout2d(0.25),
            custom2DCNN(256, 256, 2, "same"),
            nn.AvgPool2d(2),
            nn.Dropout2d(0.25),
            custom2DCNN(256, 512, 2, "same"),
            nn.AvgPool2d((2, 1)),
            nn.Dropout2d(0.25),
            custom2DCNN(512, 512, 2, "same"),
            nn.AvgPool2d((2, 1)),
            nn.Dropout2d(0.25),
            custom2DCNN(512, 512, 2, "same"),
            nn.AvgPool2d(2),
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

        self.env = EnvelopeExtractor(sample_rate=sr)

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
        a = self.logmel(x)
        b = self.env(x)

        a = self.cnn2d(a)
        b = self.cnn1d(b)

        c = torch.cat((a.squeeze(3), b), dim=1)
        x_flat = c.view(c.size(0), -1)
        z = self.fc(x_flat)
        return z
    
# class v1_1d(nn.Module):
#     def __init__(self, output_nbr, sr):
#         super(v1_1d, self).__init__()
        
#         self.logmel = LogMelSpectrogramLayer(sample_rate=sr)
#         self.env = EnvelopeExtractor(sample_rate=sr)

#         self.cnn1d = nn.Sequential(
#             custom1DCNN(1, 40, 7, "same", 2),
#             custom1DCNN(40, 40, 6, "same", 2),
#             nn.AvgPool1d(8),
#             custom1DCNN(40, 80, 5, "same", 1),
#             custom1DCNN(80, 80, 4, "same", 1),
#             nn.AvgPool1d(4),
#             custom1DCNN(80, 160, 3, "same", 1),
#             nn.AvgPool1d(4),
#             custom1DCNN(160, 160, 2, "same", 1),
#             nn.AvgPool1d(2),
#             custom1DCNN(160, 160, 2, "same", 1),
#             nn.AvgPool1d(2),
#             custom1DCNN(160, 160, 2, "same", 1),
#             nn.AvgPool1d(2),
#             custom1DCNN(160, 160, 2, "same", 1),
#             nn.AvgPool1d(7),
#         )

#         self.cnn2d = nn.Sequential(
#             custom2DCNN(1, 40, (2,3), "same"),
#             custom2DCNN(40, 40, (2,3), "same"),
#             nn.MaxPool2d((2, 1)),
#             nn.Dropout2d(0.25),
#             custom2DCNN(40, 80, (2,3), "same"),
#             custom2DCNN(80, 80, (2,3), "same"),
#             nn.MaxPool2d((2, 3)),
#             nn.Dropout2d(0.25),
#             custom2DCNN(80, 160, 2, "same"),
#             nn.MaxPool2d((2, 1)),
#             nn.Dropout2d(0.25),
#             custom2DCNN(160, 160, 2, "same"),
#             nn.MaxPool2d(2),
#             nn.Dropout2d(0.25),
#             custom2DCNN(160, 160, 2, "same"),
#             nn.MaxPool2d((2, 1)),
#             nn.Dropout2d(0.25),
#             custom2DCNN(160, 160, 2, "same"),
#             nn.MaxPool2d((2, 1)),
#             nn.Dropout2d(0.25),
#             custom2DCNN(160, 160, 2, "same"),
#             nn.MaxPool2d(2),
#             nn.Dropout2d(0.25),
#         )

#         self.fc = nn.Sequential(
#             nn.Linear(160, 80),
#             nn.ReLU(),
#             nn.Linear(80, 40),
#             nn.ReLU(),
#             nn.Linear(40, output_nbr),
#         )

#     def forward(self, x):
#         a = self.logmel(x)
#         b = self.env(x)

#         a = self.cnn2d(a)
#         b = self.cnn1d(b)
#         # c = torch.cat((a.squeeze(3), b), dim=1)
#         c = a.squeeze(3) + b
#         # c = F.normalize(c, dim=1)
#         # print(c.shape)
#         x_flat = c.view(c.size(0), -1)
#         z = self.fc(x_flat)
#         return z


class v1_1d(nn.Module):
    def __init__(self, output_nbr, sr):
        super(v1_1d, self).__init__()
        
        self.logmel = LogMelSpectrogramLayer(sample_rate=sr)
        self.env = EnvelopeExtractor(sample_rate=sr)
        self.spectral_energy = spectralEnergyExtractor()

        self.cnn1d_energy = nn.Sequential(
            custom1DCNN(1, 40, 3, "same", 1),
            nn.AvgPool1d(5),
            custom1DCNN(40, 80, 3, "same", 1),
            custom1DCNN(80, 160, 2, "same", 1),
            nn.AvgPool1d(3),
            nn.Dropout1d(0.1),
        )

        self.cnn1d = nn.Sequential(
            custom1DCNN(1, 40, 7, "same", 2),
            custom1DCNN(40, 40, 6, "same", 2),
            nn.AvgPool1d(8),
            custom1DCNN(40, 80, 5, "same", 1),
            custom1DCNN(80, 80, 4, "same", 1),
            nn.AvgPool1d(8),
            custom1DCNN(80, 160, 3, "same", 1),
            nn.AvgPool1d(4),
            custom1DCNN(160, 160, 2, "same", 1),
            nn.AvgPool1d(4),
            # custom1DCNN(160, 160, 2, "same", 1),
            # nn.AvgPool1d(2),
            # custom1DCNN(160, 160, 2, "same", 1),
            # nn.AvgPool1d(2),
            custom1DCNN(160, 160, 2, "same", 1),
            nn.AvgPool1d(7),
            nn.Dropout1d(0.1),
        )

        self.cnn2d = nn.Sequential(
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
            nn.MaxPool2d((2, 1)),
            nn.Dropout2d(0.25),
            custom2DCNN(160, 160, 2, "same"),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.25),
        )

        self.fc1 = nn.Sequential(
            nn.Linear(160, 80),
            nn.BatchNorm1d(80),
            nn.LeakyReLU(),
            # nn.Dropout1d(0.1),
        )

        self.fc2 = nn.Sequential(
            nn.Linear(160, 80),
            nn.BatchNorm1d(80),
            nn.LeakyReLU(),
            # nn.Dropout1d(0.1),
        )

        self.fc3 = nn.Sequential(
            nn.Linear(160, 80),
            nn.BatchNorm1d(80),
            nn.ReLU(),
            nn.Linear(80, output_nbr),
        )

    def forward(self, x):
        a = self.logmel(x)
        b = self.env(x)
        e = self.spectral_energy(a)

        a = self.cnn2d(a)
        b = self.cnn1d(b)
        e = self.cnn1d_energy(e)
        
        c = b.squeeze(2) + e.squeeze(2)
        fc1 = self.fc1(c)

        w = a.view(a.size(0), -1)
        fc2 = self.fc2(w)

        d = torch.cat((fc1, fc2), dim=1)
        z = d.view(d.size(0), -1)
        z = self.fc3(z)
        return z

# GOOOOD 27 09 24
# class v1_1d(nn.Module):
#     def __init__(self, output_nbr, sr):
#         super(v1_1d, self).__init__()
        
#         self.logmel = LogMelSpectrogramLayer(sample_rate=sr)
#         self.env = EnvelopeExtractor(sample_rate=sr)
#         self.spectral_energy = spectralEnergyExtractor()

#         self.cnn1d_energy = nn.Sequential(
#             custom1DCNN(1, 40, 3, "same", 1),
#             nn.AvgPool1d(5),
#             custom1DCNN(40, 80, 3, "same", 1),
#             custom1DCNN(80, 160, 2, "same", 1),
#             nn.AvgPool1d(3),
#             nn.Dropout1d(0.1),
#         )

#         self.cnn1d = nn.Sequential(
#             custom1DCNN(1, 40, 7, "same", 2),
#             custom1DCNN(40, 40, 6, "same", 2),
#             nn.AvgPool1d(8),
#             custom1DCNN(40, 80, 5, "same", 1),
#             custom1DCNN(80, 80, 4, "same", 1),
#             nn.AvgPool1d(4),
#             custom1DCNN(80, 160, 3, "same", 1),
#             nn.AvgPool1d(4),
#             custom1DCNN(160, 160, 2, "same", 1),
#             nn.AvgPool1d(2),
#             custom1DCNN(160, 160, 2, "same", 1),
#             nn.AvgPool1d(2),
#             custom1DCNN(160, 160, 2, "same", 1),
#             nn.AvgPool1d(2),
#             custom1DCNN(160, 160, 2, "same", 1),
#             nn.AvgPool1d(7),
#             nn.Dropout1d(0.1),
#         )

#         self.cnn2d = nn.Sequential(
#             custom2DCNN(1, 40, (2,3), "same"),
#             custom2DCNN(40, 40, (2,3), "same"),
#             nn.MaxPool2d((2, 1)),
#             nn.Dropout2d(0.25),
#             custom2DCNN(40, 80, (2,3), "same"),
#             custom2DCNN(80, 80, (2,3), "same"),
#             nn.MaxPool2d((2, 3)),
#             nn.Dropout2d(0.25),
#             custom2DCNN(80, 160, 2, "same"),
#             nn.MaxPool2d((2, 1)),
#             nn.Dropout2d(0.25),
#             custom2DCNN(160, 160, 2, "same"),
#             nn.MaxPool2d(2),
#             nn.Dropout2d(0.25),
#             custom2DCNN(160, 160, 2, "same"),
#             nn.MaxPool2d((2, 1)),
#             nn.Dropout2d(0.25),
#             custom2DCNN(160, 160, 2, "same"),
#             nn.MaxPool2d((2, 1)),
#             nn.Dropout2d(0.25),
#             custom2DCNN(160, 160, 2, "same"),
#             nn.MaxPool2d(2),
#             nn.Dropout2d(0.25),
#         )

#         self.fc1 = nn.Sequential(
#             nn.Linear(160 * 2, 160),
#             nn.BatchNorm1d(160),
#             nn.ReLU(),
#             nn.Dropout1d(0.1)
#         )

#         self.fc2 = nn.Sequential(
#             nn.Linear(160 * 2, 160),
#             nn.BatchNorm1d(160),
#             nn.ReLU(),
#             nn.Linear(160, 80),
#             nn.BatchNorm1d(80),
#             nn.ReLU(),
#             nn.Linear(80, output_nbr),
#         )
# # autre possibilité: env + spec )-> energie
#     def forward(self, x):
#         a = self.logmel(x)
#         b = self.env(x)
#         e = self.spectral_energy(a)

#         a = self.cnn2d(a)
#         b = self.cnn1d(b)
#         e = self.cnn1d_energy(e)
        
#         c = torch.cat((b.squeeze(2), e.squeeze(2)), dim=1)
#         c = c.view(c.size(0), -1)
#         v = self.fc1(c)

#         d = torch.cat((v, a.squeeze(2).squeeze(2)), dim=1)
#         z = d.view(d.size(0), -1)
#         z = self.fc2(z)

#         return z


class v1_mi(nn.Module):
    def __init__(self, output_nbr, sr):
        super(v1_mi, self).__init__()

        self.logmel = LogMelSpectrogramLayer(sample_rate=sr, n_mels=256)
        
        self.cnn1 = nn.Sequential(
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

        self.cnn2 = nn.Sequential(
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
            nn.Linear(160 * 2, 80),
            nn.ReLU(),
            nn.Linear(80, 40),
            nn.ReLU(),
            nn.Linear(40, output_nbr)
        )

    def forward(self, x):
        x1, x2 = torch.split(self.logmel(x), 128, dim=2)

        x1 = self.cnn1(x1)
        x2 = self.cnn2(x2)

        x = torch.cat((x1, x2), dim=1)

        x_flat = x.view(x.size(0), -1)
        z = self.fc(x_flat)
        return z
    
class v1_mi4_e(nn.Module):
    def __init__(self, output_nbr, sr):
        super(v1_mi4_e, self).__init__()

        self.logmel = LogMelSpectrogramLayer(sample_rate=sr, n_mels=512)
        self.energy = spectralEnergyExtractor()

        self.cnn_e1 = nn.Sequential(
            custom1DCNN(1, 80, 3, "same", 1),
            nn.AvgPool1d(5),
            custom1DCNN(80, 160, 3, "same", 1),
            nn.AvgPool1d(3),
            nn.Dropout1d(0.1),
        )
        
        self.cnn_e2 = nn.Sequential(
            custom1DCNN(1, 80, 3, "same", 1),
            nn.AvgPool1d(5),
            custom1DCNN(80, 160, 3, "same", 1),
            nn.AvgPool1d(3),
            nn.Dropout1d(0.1),
        )

        self.cnn_e3 = nn.Sequential(
            custom1DCNN(1, 80, 3, "same", 1),
            nn.AvgPool1d(5),
            custom1DCNN(80, 160, 3, "same", 1),
            nn.AvgPool1d(3),
            nn.Dropout1d(0.1),
        )

        self.cnn_e4 = nn.Sequential(
            custom1DCNN(1, 80, 3, "same", 1),
            nn.AvgPool1d(5),
            custom1DCNN(80, 160, 3, "same", 1),
            nn.AvgPool1d(3),
            nn.Dropout1d(0.1),
        )

        self.cnn1 = nn.Sequential(
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

        self.cnn2 = nn.Sequential(
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

        self.cnn3 = nn.Sequential(
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

        self.cnn4 = nn.Sequential(
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

        self.fcmel = nn.Sequential(
            nn.Linear(160 * 4, 320),
            nn.BatchNorm1d(320),
            nn.ReLU(),
            nn.Linear(320, 160),
            nn.BatchNorm1d(160),
            nn.ReLU(),
        )

        self.fce = nn.Sequential(
            nn.Linear(160 * 4, 320),
            nn.BatchNorm1d(320),
            nn.ReLU(),
            nn.Linear(320, 160),
            nn.BatchNorm1d(160),
            nn.ReLU(),
        )

        self.final = nn.Sequential(
            nn.Linear(160 * 2, 240),
            nn.ReLU(),
            nn.Linear(240, 40),
            nn.ReLU(),
            nn.Linear(40, output_nbr)
        )

    def forward(self, x):
        x1, x2, x3, x4 = torch.split(self.logmel(x), 128, dim=2)
        e1 = self.energy(x1)
        e2 = self.energy(x2)
        e3 = self.energy(x3)
        e4 = self.energy(x4)

        x1 = self.cnn1(x1)
        x2 = self.cnn2(x2)
        x3 = self.cnn3(x3)
        x4 = self.cnn4(x4)

        e1 = self.cnn_e1(e1)
        e2 = self.cnn_e2(e2)
        e3 = self.cnn_e3(e3)
        e4 = self.cnn_e4(e4)

        x = torch.cat((x1, x2, x3, x4), dim=1)
        e = torch.cat((e1, e2, e3, e4), dim=1)

        x_flat = x.view(x.size(0), -1)
        e_flat = e.view(e.size(0), -1)

        x = self.fcmel(x_flat)
        e = self.fce(e_flat)

        xe = torch.cat((x, e), dim=1)
        z = self.final(xe)
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
    
class v1_mi5(nn.Module):
    def __init__(self, output_nbr, sr):
        super(v1_mi5, self).__init__()

        self.logmel = LogMelSpectrogramLayer(sample_rate=sr, n_mels=640)
        
        self.cnn1 = self._create_cnn_block()
        self.cnn2 = self._create_cnn_block()
        self.cnn3 = self._create_cnn_block()
        self.cnn4 = self._create_cnn_block()
        self.cnn5 = self._create_cnn_block()

        self.fc = nn.Sequential(
            nn.Linear(160 * 5, 240),
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
        x1, x2, x3, x4, x5 = torch.split(self.logmel(x), 128, dim=2)

        x1 = self.cnn1(x1)
        x2 = self.cnn2(x2)
        x3 = self.cnn3(x3)
        x4 = self.cnn4(x4)
        x5 = self.cnn5(x5)

        x = torch.cat((x1, x2, x3, x4, x5), dim=1)

        x_flat = x.view(x.size(0), -1)
        z = self.fc(x_flat)
        return z

# ------------
class v1_mi6_env(nn.Module):
    def __init__(self, output_nbr, sr):
        super(v1_mi6_env, self).__init__()

        self.logmel = LogMelSpectrogramLayer(sample_rate=sr, n_mels=768)
        self.env = EnvelopeFollowingLayerTorchScript(n_fft=2048, hop_length=512, smoothing_factor=4)
        
        self.cnn1 = self._create_cnn_block()
        self.cnn2 = self._create_cnn_block()
        self.cnn3 = self._create_cnn_block()
        self.cnn4 = self._create_cnn_block()
        self.cnn5 = self._create_cnn_block()
        self.cnn6 = self._create_cnn_block()

        self.cnn_env = self._create_cnn_env_block()

        self.fc = nn.Sequential(
            nn.Linear(160 * 6, 320),
            nn.ReLU(),
            nn.Linear(320, 80),
            nn.ReLU(),
            nn.Linear(80, output_nbr)
        )

    def _create_cnn_env_block(self):
        return nn.Sequential(
            custom1DCNN(1, 128, 7, "same", 4),
            nn.AvgPool1d(477),
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
        x_env = self.cnn_env(x_env).unsqueeze(1)

        x1, x2, x3, x4, x5, x6 = torch.split(self.logmel(x), 128, dim=2)

        x1 = self.cnn1(x1 + x_env) 
        x2 = self.cnn2(x2 + x_env)
        x3 = self.cnn3(x3 + x_env)
        x4 = self.cnn4(x4 + x_env)
        x5 = self.cnn5(x5 + x_env)
        x6 = self.cnn6(x6 + x_env)

        x = torch.cat((x1, x2, x3, x4, x5, x6), dim=1)

        x_flat = x.view(x.size(0), -1)
        z = self.fc(x_flat)
        return z
    
class v1_mi6_env2(nn.Module):
    def __init__(self, output_nbr, sr):
        super(v1_mi6_env2, self).__init__()

        self.logmel = LogMelSpectrogramLayer(sample_rate=sr, n_mels=768)
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
            nn.ReLU(),
            nn.Linear(320, 80),
            nn.ReLU(),
            nn.Linear(80, output_nbr)
        )

    def _create_cnn_env_block(self):
        return nn.Sequential(
            custom1DCNN(1, 40, 7, 0, 4), #7144
            nn.AvgPool1d(8), #893
            custom1DCNN(40, 40, 5, 0, 3), #881
            nn.AvgPool1d(8), #110
            custom1DCNN(40, 40, 3, 0, 2), #116
            nn.AvgPool1d(4), #29
            custom1DCNN(40, 80, 2, 0, 1), #27
            nn.AvgPool1d(3), #9
            custom1DCNN(80, 160, 2, 0, 1), #7
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

        x1, x2, x3, x4, x5, x6 = torch.split(self.logmel(x), 128, dim=2)

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

# GOOOOOD 24 09 28
class v1_mi6(nn.Module):
    def __init__(self, output_nbr, sr):
        super(v1_mi6, self).__init__()

        self.logmel = LogMelSpectrogramLayer(sample_rate=sr, n_mels=768)
        
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
        x1, x2, x3, x4, x5, x6 = torch.split(self.logmel(x), 128, dim=2)

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
    
class v1_mi_lstm(nn.Module):
    def __init__(self, output_nbr, sr):
        super(v1_mi_lstm, self).__init__()

        self.logmel = LogMelSpectrogramLayer(sample_rate=sr, n_mels=512)
        
        self.cnn1 = self._create_cnn_block()
        self.cnn2 = self._create_cnn_block()
        self.cnn3 = self._create_cnn_block()
        self.cnn4 = self._create_cnn_block()

        self.lstm1 = self._create_lstm_block()
        self.lstm2 = self._create_lstm_block()
        self.lstm3 = self._create_lstm_block()
        self.lstm4 = self._create_lstm_block()

        self.fc = nn.Sequential(
            nn.Linear(160 * 4, 320),
            nn.ReLU(),
            nn.Linear(320, 80),
            nn.ReLU(),
            nn.Linear(80, output_nbr)
        )

    def _create_cnn_block(self):
        return nn.Sequential(
            custom2DCNN(1, 40, (2, 3), "same"),
            custom2DCNN(40, 40, (2, 3), "same"),
            nn.MaxPool2d((2, 1)),
            nn.Dropout2d(0.25),
            custom2DCNN(40, 80, (2, 3), "same"),
            custom2DCNN(80, 80, (2, 3), "same"),
            nn.MaxPool2d((2, 1)),
            nn.Dropout2d(0.25),
            custom2DCNN(80, 160, 2, "same"),
            nn.MaxPool2d((2, 1)),
            nn.Dropout2d(0.25),
            custom2DCNN(160, 160, 2, "same"),
            nn.MaxPool2d((2, 1)),
            nn.Dropout2d(0.25),
            custom2DCNN(160, 160, 2, "same"),
            nn.MaxPool2d((2, 1)),
            nn.Dropout2d(0.25),
            custom2DCNN(160, 160, 2, "same"),
            nn.MaxPool2d((4, 1)),
            nn.Dropout2d(0.25),
        )
    
    def _create_lstm_block(self):
        return nn.LSTM(160, hidden_size=160, batch_first=True)

    def forward(self, x):
        x1, x2, x3, x4 = torch.split(self.logmel(x), 128, dim=2)

        x1 = self.cnn1(x1).permute(0, 3, 1, 2).squeeze(2).squeeze(3)
        x2 = self.cnn2(x2).permute(0, 3, 1, 2).squeeze(2).squeeze(3)
        x3 = self.cnn3(x3).permute(0, 3, 1, 2).squeeze(2).squeeze(3)
        x4 = self.cnn4(x4).permute(0, 3, 1, 2).squeeze(2).squeeze(3)

        x1, _ = self.lstm1(x1)
        x2, _ = self.lstm2(x2)
        x3, _ = self.lstm3(x3)
        x4, _ = self.lstm4(x4)

        x1_last = x1[:, -1, :]
        x2_last = x2[:, -1, :]
        x3_last = x3[:, -1, :]
        x4_last = x4[:, -1, :]

        x = torch.cat((x1_last, x2_last, x3_last, x4_last), dim=1)
        x_flat = x.view(x.size(0), -1)
        z = self.fc(x_flat)
        return z

class v1_mi_1d(nn.Module):
    def __init__(self, output_nbr, sr):
        super(v1_mi_1d, self).__init__()

        self.logmel = LogMelSpectrogramLayer(sample_rate=sr, n_mels=256)
        self.env = EnvelopeExtractor(sample_rate=sr)

        self.cnn1d = nn.Sequential(
            custom1DCNN(1, 40, 7, "same", 2),
            custom1DCNN(40, 40, 6, "same", 2),
            nn.AvgPool1d(8),
            custom1DCNN(40, 80, 5, "same", 1),
            custom1DCNN(80, 80, 4, "same", 1),
            nn.AvgPool1d(4),
            custom1DCNN(80, 160, 3, "same", 1),
            nn.AvgPool1d(4),
            custom1DCNN(160, 160, 2, "same", 1),
            nn.AvgPool1d(2),
            custom1DCNN(160, 160, 2, "same", 1),
            nn.AvgPool1d(2),
            custom1DCNN(160, 160, 2, "same", 1),
            nn.AvgPool1d(2),
            custom1DCNN(160, 160, 2, "same", 1),
            nn.AvgPool1d(7),
            nn.Dropout1d(0.1),
        )
        
        self.cnn1 = nn.Sequential(
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

        self.cnn2 = nn.Sequential(
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
            nn.Linear(160 * 3, 120),
            nn.ReLU(),
            nn.Linear(120, 40),
            nn.ReLU(),
            nn.Linear(40, output_nbr)
        )

    def forward(self, x):
        x1, x2 = torch.split(self.logmel(x), 128, dim=2)
        x3 = self.env(x)

        x1 = self.cnn1(x1)
        x2 = self.cnn2(x2)
        x3 = self.cnn1d(x3)

        x1 = x1.view(x1.size(0), -1)
        x2 = x2.view(x2.size(0), -1)
        x3 = x3.view(x3.size(0), -1)

        w = torch.cat((x1, x2, x3), dim=1)
        z = self.fc(w)
        return z

class v1_1d_e(nn.Module):
    def __init__(self, output_nbr, sr):
        super(v1_1d_e, self).__init__()
        
        self.logmel = LogMelSpectrogramLayer(sample_rate=sr)
        self.env = EnvelopeExtractor(sample_rate=sr)
        self.spectral_energy = spectralEnergyExtractor()

        self.cnn1d_energy = nn.Sequential(
            custom1DCNN(1, 40, 3, "same", 1),
            nn.AvgPool1d(5),
            custom1DCNN(40, 80, 3, "same", 1),
            custom1DCNN(80, 160, 2, "same", 1),
            nn.AvgPool1d(3),
            nn.Dropout1d(0.1),
        )

        self.cnn1d = nn.Sequential(
            custom1DCNN(1, 40, 7, "same", 2),
            custom1DCNN(40, 40, 6, "same", 2),
            nn.AvgPool1d(8),
            custom1DCNN(40, 80, 5, "same", 1),
            custom1DCNN(80, 80, 4, "same", 1),
            nn.AvgPool1d(4),
            custom1DCNN(80, 160, 3, "same", 1),
            nn.AvgPool1d(4),
            custom1DCNN(160, 160, 2, "same", 1),
            nn.AvgPool1d(2),
            custom1DCNN(160, 160, 2, "same", 1),
            nn.AvgPool1d(2),
            custom1DCNN(160, 160, 2, "same", 1),
            nn.AvgPool1d(2),
            custom1DCNN(160, 160, 2, "same", 1),
            nn.AvgPool1d(7),
            nn.Dropout1d(0.1),
        )

        self.cnn2d = nn.Sequential(
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
            nn.MaxPool2d((2, 1)),
            nn.Dropout2d(0.25),
            custom2DCNN(160, 160, 2, "same"),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.25),
        )

        self.fc = nn.Sequential(
            nn.Linear(160, 80),
            nn.ReLU(),
            nn.Linear(80, 40),
            nn.ReLU(),
            nn.Linear(40, output_nbr),
        )

    def forward(self, x):
        a = self.logmel(x)
        b = self.env(x)
        e = self.spectral_energy(a)
        # print(e.unsqueeze(1).shape)

        a = self.cnn2d(a)
        b = self.cnn1d(b)
        e = self.cnn1d_energy(e)
        
        # c = torch.cat((a.squeeze(3), b, e), dim=1)

        c = a.squeeze(3) + b + e
        c = F.normalize(c, dim=1)
        # print(c.shape)
        x_flat = c.view(c.size(0), -1)
        z = self.fc(x_flat)
        return z


# ANCIEN
# class v1_1d(nn.Module):
#     def __init__(self, output_nbr, sr):
#         super(v1_1d, self).__init__()
        
#         self.logmel = LogMelSpectrogramLayer(sample_rate=sr)
#         self.env = EnvelopeExtractor(sample_rate=sr)

#         self.cnn1d = nn.Sequential(
#             custom1DCNN(1, 64, 8, "same", 2),
#             custom1DCNN(64, 64, 7, "same", 2),
#             nn.AvgPool1d(8),
#             custom1DCNN(64, 64, 6, "same", 1),
#             custom1DCNN(64, 64, 5, "same", 1),
#             nn.AvgPool1d(4),
#             custom1DCNN(64, 64, 4, "same", 1),
#             nn.AvgPool1d(4),
#             custom1DCNN(64, 64, 3, "same", 1),
#             nn.AvgPool1d(2),
#             custom1DCNN(64, 128, 2, "same", 1),
#             nn.AvgPool1d(2),
#             custom1DCNN(128, 128, 2, "same", 1),
#             nn.AvgPool1d(2),
#             custom1DCNN(128, 128, 2, "same", 1),
#             nn.AvgPool1d(7),
#             nn.Dropout1d(0.25),
#         )

#         self.cnn2d = nn.Sequential(
#             custom2DCNN(1, 40, (2,3), "same"),
#             custom2DCNN(40, 40, (2,3), "same"),
#             nn.MaxPool2d((2, 1)),
#             nn.Dropout2d(0.25),
#             custom2DCNN(40, 80, (2,3), "same"),
#             custom2DCNN(80, 80, (2,3), "same"),
#             nn.MaxPool2d((2, 3)),
#             nn.Dropout2d(0.25),
#             custom2DCNN(80, 160, 2, "same"),
#             nn.MaxPool2d((2, 1)),
#             nn.Dropout2d(0.25),
#             custom2DCNN(160, 160, 2, "same"),
#             nn.MaxPool2d(2),
#             nn.Dropout2d(0.25),
#             custom2DCNN(160, 160, 2, "same"),
#             nn.MaxPool2d((2, 1)),
#             nn.Dropout2d(0.25),
#             custom2DCNN(160, 160, 2, "same"),
#             nn.MaxPool2d((4, 2)),
#             nn.Dropout2d(0.25),
#         )

#         self.fc = nn.Sequential(
#             nn.Linear(160 + 128, 128),
#             nn.ELU(),
#             nn.Linear(128, 64),
#             nn.ELU(),
#             nn.Linear(64, output_nbr),
#         )

#     def forward(self, x):
#         a = self.logmel(x)
#         b = self.env(x)

#         a = self.cnn2d(a)
#         b = self.cnn1d(b)

#         c = torch.cat((a.squeeze(3), b), dim=1)
#         x_flat = c.view(c.size(0), -1)
#         z = self.fc(x_flat)
#         return z

# class one_residual(nn.Module):
#     def __init__(self, output_nbr, sr):
#         super(one_residual, self).__init__()

#         self.logmel = LogMelSpectrogramLayer(sample_rate=sr)
        
#         self.cnn_part1 = nn.Sequential(
#             custom2DCNN(1, 64, (2,3), "same"),
#             custom2DCNN(64, 64, (2,3), "same"),
#             nn.MaxPool2d((2, 1)),
#             nn.Dropout2d(0.25),
#             custom2DCNN(64, 128, (2,3), "same"),
#             custom2DCNN(128, 128, (2,3), "same"),
#             nn.MaxPool2d((2, 3)),
#             nn.Dropout2d(0.25),
#             )
        
#         self.downsample = nn.MaxPool2d((32, 5))
        
#         self.cnn_part2 = nn.Sequential(
#             custom2DCNN(128, 256, 2, "same"),
#             nn.MaxPool2d((2, 1)),
#             nn.Dropout2d(0.25),
#             custom2DCNN(256, 256, 2, "same"),
#             nn.MaxPool2d(2),
#             nn.Dropout2d(0.25),
#             custom2DCNN(256, 512, 2, "same"),
#             nn.MaxPool2d((2, 1)),
#             nn.Dropout2d(0.25),
#             custom2DCNN(512, 512, 2, "same"),
#             nn.MaxPool2d((2, 1)),
#             nn.Dropout2d(0.25),
#             custom2DCNN(512, 512, 2, "same"),
#             nn.MaxPool2d(2),
#             nn.Dropout2d(0.25),
#         )

#         self.fc_input_dim = 512 + 128
#         self.fc = nn.Sequential(
#             nn.Linear(self.fc_input_dim, 512),
#             nn.ELU(),
#             nn.Linear(512, 256),
#             nn.ELU(),
#             nn.Linear(256, output_nbr),
#         )

#     def forward(self, x):
#         x = self.logmel(x)
#         x = self.cnn_part1(x)
#         y = self.downsample(x)
#         x = self.cnn_part2(x)
#         x_flat = x.view(x.size(0), -1)
#         y_flat = y.view(y.size(0), -1)
#         z = torch.cat((x_flat, y_flat), dim=1)
#         z = self.fc(z)
#         return z

# class two_residual(nn.Module):
#     def __init__(self, output_nbr, sr):
#         super(two_residual, self).__init__()

#         self.logmel = LogMelSpectrogramLayer(sample_rate=sr)
        
#         self.cnn_part1 = nn.Sequential(
#             custom2DCNN(1, 64, (2,3), "same"),
#             custom2DCNN(64, 64, (2,3), "same"),
#             nn.MaxPool2d((2, 1)),
#             nn.Dropout2d(0.25),
#             custom2DCNN(64, 128, (2,3), "same"),
#             custom2DCNN(128, 128, (2,3), "same"),
#             nn.MaxPool2d((2, 3)),
#             nn.Dropout2d(0.25),
#             )
        
#         self.downsample1 = nn.MaxPool2d((32, 5))
        
#         self.cnn_part2 = nn.Sequential(
#             custom2DCNN(128, 256, 2, "same"),
#             nn.MaxPool2d((2, 1)),
#             nn.Dropout2d(0.25),
#             custom2DCNN(256, 256, 2, "same"),
#             nn.MaxPool2d(2),
#             nn.Dropout2d(0.25),
#             custom2DCNN(256, 512, 2, "same"),
#             nn.MaxPool2d((2, 1)),
#             nn.Dropout2d(0.25),
#         )

#         self.downsample2 = nn.MaxPool2d((4, 2))

#         self.cnn_part3 = nn.Sequential(
#             custom2DCNN(512, 512, 2, "same"),
#             nn.MaxPool2d((2, 1)),
#             nn.Dropout2d(0.25),
#             custom2DCNN(512, 512, 2, "same"),
#             nn.MaxPool2d(2),
#             nn.Dropout2d(0.25),
#         )

#         self.fc_input_dim = (512 * 2) + 128
#         self.fc = nn.Sequential(
#             nn.Linear(self.fc_input_dim, 512),
#             nn.ELU(),
#             nn.Linear(512, 256),
#             nn.ELU(),
#             nn.Linear(256, output_nbr),
#         )

#     def forward(self, x):
#         x = self.logmel(x)
#         x = self.cnn_part1(x)
#         y = self.downsample1(x)
#         x = self.cnn_part2(x)
#         z = self.downsample2(x)
#         x = self.cnn_part3(x)
#         x_flat = x.view(x.size(0), -1)
#         y_flat = y.view(y.size(0), -1)
#         z_flat = z.view(z.size(0), -1)
#         w = torch.cat((x_flat, y_flat, z_flat), dim=1)
#         w = self.fc(w)
#         return w

# class transformer(nn.Module):
#     def __init__(self, output_nbr, sr):
#         super(transformer, self).__init__()

#         self.logmel = LogMelSpectrogramLayer(sample_rate=sr)
        
#         self.cnn = nn.Sequential(
#             custom2DCNN(1, 64, (2,3), "same"),
#             custom2DCNN(64, 64, (2,3), "same"),
#             nn.MaxPool2d((2, 1)),
#             nn.Dropout2d(0.25),
#             custom2DCNN(64, 128, (2,3), "same"),
#             custom2DCNN(128, 128, (2,3), "same"),
#             nn.MaxPool2d((2, 3)),
#             nn.Dropout2d(0.25),
#             custom2DCNN(128, 256, 2, "same"),
#             nn.MaxPool2d((2, 1)), 
#             nn.Dropout2d(0.25),
#             custom2DCNN(256, 256, 2, "same"),
#             nn.MaxPool2d(2),
#             nn.Dropout2d(0.25),
#             custom2DCNN(256, 256, 2, "same"),
#             nn.MaxPool2d(2),
#             nn.Dropout2d(0.25),
#         )

#         self.transformer = nn.Transformer(d_model=256, nhead=4, num_encoder_layers=3, num_decoder_layers=3, batch_first=True)

#         self.avgpool2d = nn.AvgPool2d((4, 1)) 

#         self.fc = nn.Sequential(
#             nn.Linear(256, 128),
#             nn.ELU(),
#             nn.Linear(128, 64),
#             nn.ELU(),
#             nn.Linear(64, output_nbr),
#         )

#     def forward(self, x):
#         x = self.logmel(x)
#         x = self.cnn(x).squeeze(3)
#         x = x.permute(0, 2, 1)
#         x = self.transformer(x, x)
#         x = self.avgpool2d(x)
#         x = x.flatten(start_dim=1)
#         x = self.fc(x)
#         return x



class v1_mi4_hpss(nn.Module):
    def __init__(self, output_nbr, sr):
        super(v1_mi4_hpss, self).__init__()
        
        # Initialiser HPSS layer
        self.hpss = HPSLayer()
        
        # LogMel layer
        self.logmel = LogMelSpectrogramLayer(sample_rate=sr, n_mels=256)
        
        # CNN pour harmonic et percussive
        self.cnn1 = self._create_cnn_block()
        self.cnn2 = self._create_cnn_block()
        self.cnn3 = self._create_cnn_block()
        self.cnn4 = self._create_cnn_block()

        # Fully connected layer
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
        harmonic, percussive = self.hpss(x)

        mel_harmonic = self.logmel(harmonic)
        mel_percussive = self.logmel(percussive)

        x1, x2 = torch.split(mel_harmonic, 128, dim=2)
        x3, x4 = torch.split(mel_percussive, 128, dim=2)

        x1 = self.cnn1(x1)
        x2 = self.cnn2(x2)
        x3 = self.cnn3(x3)
        x4 = self.cnn4(x4)

        x = torch.cat((x1, x2, x3, x4), dim=1)
        x_flat = x.view(x.size(0), -1)
        z = self.fc(x_flat)
        return z