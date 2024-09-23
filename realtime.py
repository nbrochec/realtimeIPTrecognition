#############################################################################
# test_pyaudio.py 
# Nicolas Brochec
# TOKYO UNIVERSITY OF THE ARTS
# 東京藝術大学音楽音響創造科
# ERC Reach
# GNU General Public License v3.0
#############################################################################
# Code description:
# Run a personnal model in real time.
#############################################################################

import argparse
import pyaudio, os, librosa, time, pythonosc, math, time, glob
import torch
import torchaudio
from pythonosc import udp_client, osc_message_builder
from pythonosc.dispatcher import Dispatcher
import numpy as np
import torch.nn.functional as nnf

import yaml
from yaml import load
try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader

from utils import *
from models import *

# Arguments
def parse_arguments():
    parser = argparse.ArgumentParser(description='This script run the model in real time.')
    parser.add_argument('--input', type=int, required=True, help='Audio Device ID.')
    parser.add_argument('--channel', type=int, required=True, help='Channel of Audio Device to listen to.')
    parser.add_argument('--device', type=str, default='cpu', help='GPU device.')
    parser.add_argument('--gpu', type=int, default=0, help='GPU device number.')
    parser.add_argument('--name', type=str, required=True, help='Name of the project.')
    parser.add_argument('--buffer_size', type=int, default=256, help='Specify audio buffer size.')
    parser.add_argument('--moving_average', type=int, default=5, help='Window size for smoothing predictions with a moving average.')
    parser.add_argument('--port', type=int, default=5005, help='Specify UDP port.')
    return parser.parse_args()

args = parse_arguments()
GetDevice.get_device(args)

path_to_runs = os.path.join(os.getcwd(), 'runs')

list_of_yaml = glob.glob(os.path.join(path_to_runs, f'{args.name}_*/{args.name}_*.yaml'))
latest_yaml = max(list_of_yaml, key=os.path.getctime)

list_of_models = glob.glob(os.path.join(path_to_runs, f'{args.name}_*/{args.name}_*.pth'))
latest_model = max(list_of_models, key=os.path.getctime)

yaml_name = os.path.splitext(os.path.basename(latest_yaml))[0]
model_name = os.path.splitext(os.path.basename(latest_model))[0]

if yaml_name != model_name:
    raise ValueError("The YAML and PTH files do not correspond to the same date.")

yaml_file = open(latest_yaml, 'r')
yaml_dict = yaml.load(yaml_file, Loader=Loader)

ckpt = torch.load(latest_model, map_location=args.device)

model_loader = LoadModel()
model = model_loader.get_model(yaml_dict['Model'], int(yaml_dict['Number of Classes'], int(yaml_dict['Sampling Rate'])))
model.load_state_dict(ckpt)
model.eval()

audioFlux = pyaudio.PyAudio() 
SR_ORIGINAL = int(audioFlux.get_device_info_by_index(args.input)['defaultSampleRate'])
SR_TARGET = yaml_dict['Sampling Rate']
NUM_CLASSES = yaml_dict['Number of Classes']
SMOOTH_WINDOW = 10

pred_buffer = PredictionBuffer(NUM_CLASSES, SMOOTH_WINDOW)
sender = SendOSCMessage(args)
cumulativeAudio = torch.zeros((1, 1, SEGMENT_LENGTH))


def callback(in_data, frame_count, time_info, flag):
    global cumulativeAudio

    audioSample = torch.frombuffer(in_data, dtype=torch.float32)
    audioSample = Resample.resample(audioSample, SR_ORIGINAL, SR_TARGET)

    concat = torch.concatenate((cumulativeAudio, audioSample.unsqueeze(0).unsqueeze(0)), axis=2)

    if concat.shape[2] >= SEGMENT_LENGTH:
        concat = concat[:,:,-SEGMENT_LENGTH:]
        audioON = torch.sum(torch.abs(audioSample))

        if audioON > 0:
            print('Now running!')
            out_prob = MakeInference.make_inference(model, concat)
            pred_buffer.update_buffer(out_prob)
            smooth_average = torch.mean(pred_buffer.get_buffer(), dim=0)
            pred = torch.argmax(smooth_average).item()

            sender.send_message(pred=pred)
        else:
            print('Waiting for audio...')
            time.sleep(0.5)

    cumulativeAudio = concat[:,:,-SEGMENT_LENGTH:]

    return None, pyaudio.paContinue


audioStream = audioFlux.open(format=pyaudio.paFloat32,
                 channels=args.channel,
                 rate=SR_ORIGINAL,
                 output=False,
                 input=True,
                 input_device_index=args.input, # Change le numéro de périphérique audio ici (il faut lancer pyaudio-check.py pour trouver le bon numéro)
                 stream_callback=callback,
                 frames_per_buffer=args.buffer_size)

if __name__ == '__main__':
    audioStream.start_stream()

    while audioStream.is_active():
        time.sleep(0.25)
    audioStream.close()
    pyaudio.terminate()

