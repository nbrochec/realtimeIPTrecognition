import torch
import os
import argparse

from models import *
from utils import *

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='This script launches the training process.')
    parser.add_argument('--device', type=str, default='cuda', help='Specify the hardware on which computation should be performed.')
    parser.add_argument('--run_name', type=str, help='Name of the run.', required=True)
    return parser.parse_args()

def get_run_dir(run_name):
    """Create runs and the current run directories."""
    cwd = os.getcwd()
    runs = os.path.join(cwd, 'runs', run_name)
    return runs

def get_ts_file(run_dir):
    for file_name in os.listdir(run_dir):
        if file_name.endswith('.ts'):
            return os.path.join(run_dir, file_name)

if __name__ == '__main__':
    args = parse_arguments()

    run_dir = get_run_dir(args.run_name)
    model_path = get_ts_file(run_dir)

    model_gpu = torch.jit.load(model_path, map_location=args.device)
    model_cpu = model_gpu.to('cpu')

    model_cpu_path = os.path.join(run_dir, f'cpu_{os.path.basename(model_path)}')
    torch.jit.save(model_cpu, model_cpu_path)
    print(f'Model has been transfered to CPU and save at {os.path.relpath(model_path)}')




    