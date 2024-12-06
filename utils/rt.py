#############################################################################
# rt.py 
# Nicolas Brochec
# TOKYO UNIVERSITY OF THE ARTS
# 東京藝術大学音楽音響創造科
# ERC Reach
# GNU General Public License v3.0
#############################################################################
# Code description:
# Implement real time utility functions
#############################################################################

import torch
import time
import torchaudio
from pythonosc import udp_client, osc_message_builder
import torch.nn.functional as F

class Resample:
    @staticmethod
    def resample(tensor, original_sr, target_sr):
        return torchaudio.functional.resample(tensor, original_sr, target_sr)

class Latency:
    @staticmethod
    def measure(start_time):
        return round((time.time() - start_time) * 1000)
    
class SendOSCMessage:
    def __init__(self, args):
        self.ip = "127.0.0.1" #localhost
        self.port = args.port
        self.client = udp_client.SimpleUDPClient(self.ip, self.port)

    def send_message(self, pred, latency):
        classMSG = osc_message_builder.OscMessageBuilder(address= '/class')
        classMSG.add_arg(pred, arg_type='i')
        classMSG = classMSG.build()

        totalDelayMSG = osc_message_builder.OscMessageBuilder(address= '/delay')
        totalDelayMSG.add_arg(latency, arg_type='f')
        totalDelayMSG = totalDelayMSG.build()

        self.client.send(classMSG)
        self.client.send(totalDelayMSG)

class PredictionBuffer:
    def __init__(self, num_classes, window_size):
        self.num_classes = num_classes
        self.window_size = window_size
        self.bufferPROB = torch.zeros((window_size, num_classes))

    def update_buffer(self, new_prob):
        self.bufferPROB[:-1, :] = self.bufferPROB[1:, :].clone()
        self.bufferPROB[-1, :] = new_prob

    def get_buffer(self):
        return self.bufferPROB
    
class MakeInference:
    @staticmethod
    def make_inference(model, input_data):
        out = model(input_data)
        prob = F.softmax(out, dim=1).squeeze().detach().cpu()
        return prob

