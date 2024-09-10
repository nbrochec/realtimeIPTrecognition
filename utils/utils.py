#############################################################################
# utils.py 
# Nicolas Brochec
# TOKYO UNIVERSITY OF THE ARTS
# 東京藝術大学音楽音響創造科
# ERC Reach
# GNU General Public License v3.0
#############################################################################
# Code description:
# Implement global utility functions
#############################################################################

import os, csv, yaml
import torch, torchaudio, random

import pandas as pd

from glob import glob
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader
from externals.pytorch_balanced_sampler.sampler import SamplerFactory

class DirectoryManager:
    @staticmethod
    def ensure_dir_exists(directory):
        """
        Ensures that the directory exists. If not, creates it.
        """
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f'{directory} has been created.')

class DatasetSplitter:
    @staticmethod
    def split_train_validation(base_dir, destination='data/dataset/', train_dir='train', test_dir='test', val_ratio=0.2, val_split='train', name='title'):
        """
        Splits the dataset into training, validation, and test sets, and saves the information in a CSV file.

        Parameters
        ----------
        base_dir : str
            The base directory containing the data.
        train_dir : str
            The name of the training directory.
        test_dir : str
            The name of the test directory.
        val_ratio : float
            The ratio of the validation set to the total training dataset.
        val_split : str
            On which dataset the validation split will be made.
        csv_filename : str
            The name of the output CSV file.
        """
        train_path = os.path.join(base_dir, train_dir)
        test_path = os.path.join(base_dir, test_dir)
        csv_filename = f'{name}_dataset_split.csv'

        csv_path = os.path.join(destination, csv_filename)

        if val_split != 'train' and val_split != 'test':
            raise ValueError("val_split must be either 'train' or 'test'.")

        with open(csv_path, mode='w', newline='') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(['file_path', 'label', 'set'])

            # Process train directory
            for root, dirs, files in tqdm(os.walk(train_path), desc='Process training audio files.'):
                label = os.path.basename(root)
                all_files = [os.path.join(root, f) for f in files if f.lower().endswith(('.wav', '.aiff', '.aif', '.mp3'))]

                if val_split == 'train':
                    # Split into train and validation sets
                    num_files = len(all_files)
                    num_val = int(num_files * val_ratio)
                    val_files = random.sample(all_files, num_val)
                    train_files = list(set(all_files) - set(val_files))

                    for file in train_files:
                        writer.writerow([file, label, 'train'])

                    for file in val_files:
                        writer.writerow([file, label, 'val'])
                else:
                    train_files = list(set(all_files))
                    for file in train_files:
                        writer.writerow([file, label, 'train'])

            # Process test directory
            for root, dirs, files in tqdm(os.walk(test_path), desc='Process test audio files.'):
                label = os.path.basename(root)
                all_files = [os.path.join(root, f) for f in files if f.lower().endswith(('.wav', '.aiff', '.aif', '.mp3'))]

                if val_split == 'test':
                    # Split into test and validation sets
                    num_files = len(all_files)
                    num_val = int(num_files * val_ratio)
                    val_files = random.sample(all_files, num_val)
                    test_files = list(set(all_files) - set(val_files))

                    for file in test_files:
                        writer.writerow([file, label, 'test'])

                    for file in val_files:
                        writer.writerow([file, label, 'val'])
                else:
                    test_files = list(set(all_files))
                    for file in test_files:
                        writer.writerow([file, label, 'test'])

        print(f"CSV file '{csv_filename}' created successfully in {base_dir}.")

class DatasetValidator:
    @staticmethod
    def validate_labels(csv_file):
        """
        Validates that the train, test, and val sets have the same unique labels.

        Parameters
        ----------
        csv_file : str
            Path to the CSV file containing dataset information.

        Raises
        ------
        ValueError
            If the labels in the train, test, and val sets do not match in number or name.
        """
        data = pd.read_csv(csv_file)

        train_labels = set(data[data['set'] == 'train']['label'].unique())
        test_labels = set(data[data['set'] == 'test']['label'].unique())
        val_labels = set(data[data['set'] == 'val']['label'].unique())

        if not (train_labels == test_labels == val_labels):
            raise ValueError("Mismatch in labels between train, test, and val sets.")
        
        print("Label validation passed: All sets have the same labels.")

    def get_num_classes_from_csv(csv_file):
        data = pd.read_csv(csv_file)
        return len(data['label'].unique())


class ProcessDataset:
    def __init__(self, set_type, csv_path, target_sr, segment_overlap, segment_length, silence_threshold=1e-4, min_silence_len=0.1):
        """
        Initialize the ProcessDataset class.

        Parameters
        ----------
        set_type : str
            The type of dataset to process ('train', 'test', or 'val').
        csv_path : str
            Path to the CSV file containing file paths, labels, and set information.
        target_sr : int
            Target sampling rate for audio files.
        segment_length : int
            The length of each audio segment to be extracted.
        silence_threshold : float
            Threshold below which sound is considered silence.
        min_silence_len : float
            Minimum length of silence to remove.
        """
        self.set_type = set_type
        self.csv_path = csv_path
        self.target_sr = target_sr
        self.segment_length = segment_length
        self.silence_threshold = silence_threshold
        self.min_silence_len = min_silence_len
        self.segment_overlap = segment_overlap # implement rollof 

        self.data = pd.read_csv(self.csv_path)
        
        self.data = self.data[self.data['set'] == self.set_type]
        
        self.label_map = {label: idx for idx, label in enumerate(sorted(self.data['label'].unique()))}

        self.X = []
        self.y = []

        self.process_all_files()

    def remove_silence(self, waveform):
        """
        Remove silence from the audio waveform.
        """
        min_silence_samples = int(self.min_silence_len * self.target_sr)
        amplitude = torch.sqrt(torch.mean(waveform**2, dim=0))
        non_silent_indices = torch.where(amplitude > self.silence_threshold)[0]

        if len(non_silent_indices) == 0:
            return waveform

        start = max(0, non_silent_indices[0] - min_silence_samples)
        end = min(waveform.shape[1], non_silent_indices[-1] + min_silence_samples)
        return waveform[:, start:end]
    
    def process_all_files(self):
        """
        Process all audio files and store them in X and y.
        """
        for _, row in self.data.iterrows():
            file_path = row['file_path']
            label = row['label']
            label = self.label_map[label]

            waveform, original_sr = torchaudio.load(file_path)

            if original_sr != self.target_sr:
                waveform = torchaudio.transforms.Resample(orig_freq=original_sr, new_freq=self.target_sr)(waveform)

            waveform = self.remove_silence(waveform)
            num_samples = waveform.size(1)

            if self.segment_overlap == True and self.set_type == 'train':
                for i in range(0, num_samples, self.segment_length//2):
                    if i + self.segment_length <= num_samples:
                        segment = waveform[:, i:i + self.segment_length]
                    else:
                        segment = torch.zeros((waveform.size(0), self.segment_length))
                        segment[:, :num_samples - i] = waveform[:, i:]

                    self.X.append(segment)
                    self.y.append(label)
            else:
                for i in range(0, num_samples, self.segment_length):
                    if i + self.segment_length <= num_samples:
                        segment = waveform[:, i:i + self.segment_length]
                    else:
                        segment = torch.zeros((waveform.size(0), self.segment_length))
                        segment[:, :num_samples - i] = waveform[:, i:]

                    self.X.append(segment)
                    self.y.append(label)

        self.X = torch.stack(self.X)
        self.y = torch.tensor(self.y)

    def get_data(self):
        return TensorDataset(self.X, self.y).to(self.device)

class BalancedDataLoader:
    """
    A class for creating a balanced DataLoader from a PyTorch dataset.

    Parameters
    ----------
    dataset : Dataset
        The PyTorch dataset.
    """
    def __init__(self, dataset, device):
        self.dataset = dataset
        self.num_classes = self.get_num_classes()
        self.batch_size = self.num_classes
        self.device = device

        all_targets = [dataset[i][1].unsqueeze(0) if dataset[i][1].dim() == 0 else dataset[i][1] for i in range(len(dataset))]
        all_targets = torch.cat(all_targets)

        class_idxs = [[] for _ in range(self.num_classes)]

        for i in range(self.num_classes):
            indexes = torch.nonzero(all_targets == i).squeeze()
            class_idxs[i] = indexes.tolist()

        total_samples = len(self.dataset)
        n_batches = total_samples // self.batch_size

        self.batch_sampler = SamplerFactory().get(
            class_idxs=class_idxs,
            batch_size=self.batch_size,
            n_batches=n_batches,
            alpha=1,
            kind='fixed'
        )

    def get_num_classes(self):
        """
        Determines the number of unique classes in the dataset.
        """
        all_labels = [label.item() for label in self.dataset.tensors[1]]
        unique_classes = set(all_labels)
        num_unique_classes = len(unique_classes)
        return num_unique_classes
    
    def custom_collate_fn(self, batch):
        segments = []
        labels = []
        
        for segs, lbls in batch:
            segments.extend(segs)
            labels.extend(lbls)

        segments_tensor = torch.stack(segments).to(self.device)
        labels_tensor = torch.tensor(labels).to(self.device)

        return segments_tensor, labels_tensor

    def get_dataloader(self):
        """
        Returns a DataLoader with the balanced batch sampler.
        """
        return DataLoader(
            self.dataset,
            batch_sampler=self.batch_sampler,
        )

class PrepareData:
    """
    Prepare datasets.
    """
    def __init__(self, args, csv_file_path, seg_len):
        self.args = args
        self.csv = csv_file_path
        self.seg_len = seg_len
        self.device = args.device

    def prepare(self):
        num_classes = DatasetValidator.get_num_classes_from_csv(self.csv)
        train_dataset = ProcessDataset('train', self.csv, self.args.sr, self.args.segment_overlap, self.seg_len)
        test_dataset = ProcessDataset('test', self.csv, self.args.sr, self.args.segment_overlap, self.seg_len)
        val_dataset = ProcessDataset('val', self.csv, self.args.sr, self.args.segment_overlap, self.seg_len)

        train_loader = BalancedDataLoader(train_dataset.get_data(), self.device).get_dataloader()
        test_loader = DataLoader(test_dataset.get_data(), batch_size=64)
        val_loader = DataLoader(val_dataset.get_data(), batch_size=64)
        
        print('Data successfully loaded into DataLoaders.')

        return train_loader, test_loader, val_loader, num_classes

class SaveResultsToDisk:
    @staticmethod
    def get_class_names(csv_file_path):
        data = pd.read_csv(csv_file_path)
        data = data[data['set'] == 'train']
        label_map = {label: idx for idx, label in enumerate(sorted(data['label'].unique()))}

        return label_map

    @staticmethod
    def save_to_disk(args, stacked_metrics, cm, date, time, csv_file_path):
        """
        Save the results to disk as a CSV file.
        """

        label_map = SaveResultsToDisk.get_class_names(csv_file_path)
        labels = sorted(label_map.keys())

        log_dir = os.path.join(os.getcwd(), 'logs', args.name)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir) 

        csv_path = os.path.join(log_dir, f'results.csv')

        accuracy, precision, recall, f1, loss = stacked_metrics.tolist()

        write_header = not os.path.exists(csv_path)

        with open(csv_path, mode='a', newline='') as csv_file:
            writer = csv.writer(csv_file)

            if write_header:
                writer.writerow([
                    'Date', 'Time', 'Model Name', 'Sample Rate', 'Segment Overlap',
                    'Fmin', 'Learning Rate', 'Epochs', 'Augmentations', 'Early Stopping',
                    'Reduce LR on Plateau', 'Accuracy', 'Precision', 'Recall', 'F1 Score', 'Loss'
                ])

            writer.writerow([
                date, time, args.name, args.sr, args.segment_overlap,
                args.fmin, args.lr, args.epochs, args.augment,
                args.early_stopping, args.reduceLR, accuracy, precision,
                recall, f1, loss
            ])

        cm_path = os.path.join(log_dir, f'{date}_{time}_cm.csv')
        cm_np = cm.cpu().numpy()
        df_cm = pd.DataFrame(cm_np, index=labels, columns=labels)
        df_cm.to_csv(cm_path)

        print(f'Results saved to {csv_path}')

class SaveYAML:
    @staticmethod
    def save_to_disk(args, num_classes):
        name = args.name
        cwd = os.getcwd()
        path_to_run = os.path.join(cwd, 'runs', name)

        if not os.path.exists(os.path.join(cwd, 'runs')):
            os.makedirs(os.path.join(cwd, 'runs'))

        if not os.path.exists(path_to_run):
            os.makedirs(path_to_run)

        current_config = {'Name':args.name, 'Model': args.config , 'Sampling Rate':args.sr,'Segment Overlap':args.segment_overlap,
                      'Fmin':args.fmin, 'Learning Rate': args.lr, 'Epochs': args.epochs, 'Augmentations':args.augment,
                      'Early Stopping':args.early_stopping,'Reduce LR on Plateau':args.reduceLR, 'Number of Classes':num_classes}

        yaml_file = os.path.join(path_to_run, f'{name}.yaml')

        if not os.path.exists(yaml_file):   
            with open(yaml_file, 'w') as file:
                yaml.dump(current_config, file, default_flow_style=False)

class GetDevice:
    @staticmethod
    def get_device(args):
        """Automatically select the device and move everything to it."""
        device_name = args.device
        gpu = args.gpu
        
        if device_name != 'cpu':
            device = torch.device(f'{device_name}:{gpu}')
            print(f'This script uses {device_name}:{gpu} as the torch device.')
        else:
            device = torch.device('cpu')
            print(f'This script uses CPU as the torch device.')
        
        return device