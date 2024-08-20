#############################################################################
# CNN_arch.py 
# Nicolas Brochec
# TOKYO UNIVERSITY OF THE ARTS
# 東京藝術大学音楽音響創造科
# ERC Reach
# MIT license
#############################################################################
# Code description:
# Different architectures for the RT-IPT-R
#############################################################################

import torch
import torch.nn as nn
import torchaudio.transforms as T
import torch.nn.functional as F
import torchaudio.functional as Faudio

class LogMelSpectrogramLayer(nn.Module):
    def __init__(self, sample_rate=24000, n_fft=2048, win_length=None, hop_length=512, n_mels=128, f_min=150, f_max=None):
        super(LogMelSpectrogramLayer, self).__init__()
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.win_length = win_length if win_length is not None else n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.f_min = f_min
        self.f_max = f_max if f_max is not None else sample_rate / 2
        
        self.mel_scale = T.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=self.n_fft,
            win_length=self.win_length,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
            f_min=self.f_min,
            f_max=self.f_max,
            power=2.0
        )

        self.amplitude_to_db = T.AmplitudeToDB(stype='power', top_db=80.0)

    def forward(self, x):
        x = self.mel_scale(x)
        x = self.amplitude_to_db(x)
        x = F.normalize(x, dim=2)
        x = F.normalize(x, dim=3)
        return x

class EnvelopeExtractor(nn.Module):
    def __init__(self, sample_rate=24000, cutoff_freq=10, Q=0.707):
        super(EnvelopeExtractor, self).__init__()
        self.sr = sample_rate
        self.cutoff = cutoff_freq
        self.Q = Q

    def forward(self, x):
        x = torch.abs(x)
        x = Faudio.highpass_biquad(x, sample_rate=self.sr, cutoff_freq=self.cutoff, Q=self.Q)
        return x

class custom2DCNN(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size, padding):
        super(custom2DCNN, self).__init__()
        self.conv = nn.Conv2d(in_channels=input_channels, out_channels=output_channels, kernel_size=kernel_size, stride=1, padding=padding)
        self.batch = nn.BatchNorm2d(output_channels)
        self.activ = nn.LeakyReLU()
        self.drop = nn.Dropout2d(0.25)

    def forward(self, x):
        x = self.conv(x)
        x = self.batch(x)
        x = self.activ(x)
        x = self.drop(x)
        return x

class custom1DCNN(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size, padding, dilation):
        super(custom1DCNN, self).__init__()
        self.conv = nn.Conv1d(in_channels=input_channels, out_channels=output_channels,kernel_size=kernel_size, padding=padding, dilation=dilation)
        self.batch = nn.BatchNorm1d(output_channels)
        self.activ = nn.GELU()
        self.avg = nn.AvgPool1d(4)
    
    def forward(self,x):
        x = self.conv(x)
        x = self.batch(x)
        x = self.activ(x)
        # x = self.drop(x)
        return x

class v1(nn.Module):
    def __init__(self, output_nbr):
        super(v2, self).__init__()

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
            nn.MaxPool2d((4, 1)),
        )

        self.fc = nn.Sequential(
            nn.Linear(160, output_nbr),
        )

    def forward(self, x):
        x = self.logmel(x)
        x = self.cnn(x)
        x = self.fc(x)
        return x

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
        x = self.fc(x)
        return x

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
        
        self.downsample = nn.MaxPool2d((5, 32))
        
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
        z_flat = y.view(z.size(0), -1)

        w = torch.cat((x_flat, y_flat, z_flat), dim=1)
        return w

class transformer(nn.Module):
    def __init__(self, output_nbr):
        super(transformer, self).__init__()

        self.logmel = LogMelSpectrogramLayer(sample_rate=24000)
        
        self.cnn = nn.Sequential(
            custom2DCNN(1, 64, (2,3), "same"),
            custom2DCNN(64, 64, (2,3), "same"),
            nn.MaxPool2d((2, 1)), #64, 15

            custom2DCNN(64, 128, (2,3), "same"),
            custom2DCNN(128, 128, (2,3), "same"),
            nn.MaxPool2d((2, 3)), #32, 5

            custom2DCNN(128, 256, 2, "same"),
            nn.MaxPool2d((2, 1)), #16, 5

            custom2DCNN(256, 256, 2, "same"),
            nn.MaxPool2d(2), #8, 2

            custom2DCNN(256, 512, 2, "same"),
            nn.MaxPool2d(2), #4
        )

        self.transformer = nn.Transformer(512, 8, 6, 6)

        self.maxpool2d = nn.MaxPool2d((4, 1)) 

        self.fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.ELU(),
            nn.Linear(128, 128),
            nn.ELU(),
            nn.Linear(128, output_nbr),
        )

    def forward(self, x):
        x = self.logmel(x)
        x = self.cnn(x)

        x = x.permute(2, 0, 1)

        x = self.transformer(x, x)

        x = self.maxpool2d(x)

        x = self.fc(x)
        return x

