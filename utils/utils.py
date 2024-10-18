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
import torch, torchaudio, random, librosa

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
import numpy as np

from glob import glob
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader
from externals.pytorch_balanced_sampler.sampler import SamplerFactory

from utils.augmentation import AudioOfflineTransforms

from sklearn.model_selection import train_test_split

class DirectoryManager:
    @staticmethod
    def ensure_dir_exists(directory):
        """Ensures that the directory exists. If not, creates it."""
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f'{directory} has been created.')

class DatasetSplitter:
    @staticmethod
    def split_train_validation(base_dir, destination='data/dataset/', train_dir='train', test_dir='test', val_dir=None, val_ratio=0.2, val_split='train', name='title'):
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
        val_dir : str
            The name of the val directory.
        val_ratio : float
            The ratio of the validation set to the total training dataset.
        val_split : str
            On which dataset the validation split will be made.
        csv_filename : str
            The name of the output CSV file.
        """
        train_path = os.path.join(base_dir, train_dir)
        test_path = os.path.join(base_dir, test_dir)

        if val_dir is not None:
            val_path = os.path.join(base_dir, val_dir)
        else:
            val_path = None
            
        csv_filename = f'{name}_dataset_split.csv'

        csv_path = os.path.join(destination, csv_filename)

        if val_split != 'train' and val_split != 'test':
            raise ValueError("val_split must be either 'train' or 'test'.")
        
        if val_path is None:
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
            
        if val_path is not None:
            with open(csv_path, mode='w', newline='') as csv_file:
                writer = csv.writer(csv_file)
                writer.writerow(['file_path', 'label', 'set'])

                for root, dirs, files in tqdm(os.walk(train_path), desc='Process training audio files.'):
                    label = os.path.basename(root)
                    all_files = [os.path.join(root, f) for f in files if f.lower().endswith(('.wav', '.aiff', '.aif', '.mp3'))]

                    train_files = list(set(all_files))
                    for file in train_files:
                        writer.writerow([file, label, 'train'])

                for root, dirs, files in tqdm(os.walk(test_path), desc='Process test audio files.'):
                    label = os.path.basename(root)
                    all_files = [os.path.join(root, f) for f in files if f.lower().endswith(('.wav', '.aiff', '.aif', '.mp3'))]
                    
                    test_files = list(set(all_files))
                    for file in test_files:
                        writer.writerow([file, label, 'test'])

                for root, dirs, files in tqdm(os.walk(val_path), desc='Process val audio files.'):
                    label = os.path.basename(root)
                    all_files = [os.path.join(root, f) for f in files if f.lower().endswith(('.wav', '.aiff', '.aif', '.mp3'))]
                    
                    val_files = list(set(all_files))
                    for file in val_files:
                        writer.writerow([file, label, 'val'])           

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
    
    def get_classnames_from_csv(csv_file):
        data = pd.read_csv(csv_file)
        return sorted(data['label'].unique())

class ProcessDataset:
    def __init__(self, set_type, csv_path, args, segment_length):
        """
        Initialize the ProcessDataset class.

        Parameters
        ----------
        set_type : str
            The type of dataset to process ('train', 'test', or 'val').
        csv_path : str
            Path to the CSV file containing file paths, labels, and set information.
        args : Arguments
            Arguments object containing settings for augmentations and other configurations.
        segment_length : int
            The length of each audio segment to be extracted.
        """
        self.set_type = set_type
        self.csv_path = csv_path
        self.target_sr = args.sr
        self.segment_length = segment_length
        # self.silence_threshold = silence_threshold
        # self.min_silence_len = min_silence_len
        self.segment_overlap = args.segment_overlap
        self.padding = args.padding
        self.args = args
        self.offline_aug = args.offline_augment
        
        #self.offline_aug = True

        self.data = pd.read_csv(self.csv_path)
        
        self.data = self.data[self.data['set'] == self.set_type]
        
        self.label_map = {label: idx for idx, label in enumerate(sorted(self.data['label'].unique()))}

        self.X = []
        self.y = []

        self.process_all_files()

    def remove_silence(self, waveform):
        """Remove silence from the audio waveform."""
        wav = waveform.detach().cpu().numpy()
        wav = librosa.effects.trim(wav)
        return torch.tensor(wav[0])
    
    def pad_waveform(self, waveform, target_length):
        """Add silence to the waveform to match the target length."""
        extra_length = target_length - waveform.size(1)
        if extra_length > 0:
            silence = torch.zeros((waveform.size(0), extra_length))
            waveform = torch.cat((waveform, silence), dim=1)
        return waveform
    
    def process_segment(self, waveform):
        """Process the waveform by dividing it into segments with or without overlap."""
        segments = []
        num_samples = waveform.size(1)

        for i in range(0, num_samples, self.segment_length if not self.segment_overlap else self.segment_length // 2):
            if i + self.segment_length <= num_samples:
                segment = waveform[:, i:i + self.segment_length]
            else:
                if self.padding == 'full':
                    valid_length = num_samples - i
                    segment = torch.zeros((waveform.size(0), self.segment_length))
                    segment[:, :valid_length] = waveform[:, i:i + valid_length]
            segments.append(segment)

        return segments
    
    def process_all_files(self):
        """Process all audio files and store them in X and y."""
        augmenter = AudioOfflineTransforms(self.args) if self.offline_aug else None
        if self.offline_aug and self.set_type == 'train':
            print(f'self offline augmentations: {self.offline_aug}')  

        for _, row in tqdm(self.data.iterrows()):
            file_path, label_name = row['file_path'], row['label']
            label = self.label_map[label_name]

            waveform, original_sr = torchaudio.load(file_path)

            if original_sr != self.target_sr:
                waveform = torchaudio.transforms.Resample(orig_freq=original_sr, new_freq=self.target_sr)(waveform)

            if label_name != 'silence':
                waveform = self.remove_silence(waveform)

            if waveform.shape[0] == 2:
                waveform = waveform[0, :].unsqueeze(0)

            if self.padding == 'minimal' and waveform.size(1) < self.segment_length:
                waveform = self.pad_waveform(waveform, self.segment_length)

            segments = self.process_segment(waveform)

            for segment in segments:
                if augmenter and self.set_type == 'train':
                    aug1, aug2, aug3 = augmenter(segment)
                    self.X.extend([aug1, aug2, aug3])
                    self.y.extend([label] * 3)

                self.X.append(segment)
                self.y.append(label)

        self.X = torch.stack(self.X)
        self.y = torch.tensor(self.y)
    
    def get_data(self):
        return TensorDataset(self.X, self.y)

class BalancedDataLoader:
    """
    A class for creating a balanced DataLoader from a PyTorch dataset.

    Parameters
    ----------
    dataset : Dataset
        The PyTorch dataset.
    args : Arguments
        Arguments object containing settings and other configurations.
    """
    def __init__(self, dataset, args):
        self.dataset = dataset
        self.num_classes = self.get_num_classes()
        # self.device = device
        self.args = args
        # self.batch_size = args.batch_size
        self.batch_size = self.args.batch_size

        all_targets = [dataset[i][1].unsqueeze(0) if dataset[i][1].dim() == 0 else dataset[i][1] for i in range(len(dataset))]
        all_targets = torch.cat(all_targets)

        class_idxs = [[] for _ in range(self.num_classes)]
        for i in range(self.num_classes):
            indexes = torch.nonzero(all_targets == i, as_tuple=True)
            if indexes[0].numel() > 0:
                class_idxs[i] = indexes[0].tolist()
            else:
                print(f"Class {i} has no indices")

        total_samples = len(self.dataset)
        n_batches = total_samples // self.batch_size

        class_counts = [0] * self.num_classes
        for i in range(len(self.dataset)):
            _, label = self.dataset[i]
            if label.dim() == 0:
                label = label.item()
            else:
                label = label.argmax().item()
            if 0 <= label < self.num_classes:
                class_counts[label] += 1

        print(f"Class distribution: {class_counts}")

        self.batch_sampler = SamplerFactory().get(
            class_idxs=class_idxs,
            batch_size=self.batch_size,
            n_batches=n_batches,
            alpha=1,
            kind='fixed'
        )

    def get_num_classes(self):
        """ Determines the number of unique classes in the dataset. """
        all_labels = [label.item() for label in self.dataset.tensors[1]]
        unique_classes = set(all_labels)
        num_classes = len(unique_classes)
        print(f"Unique classes detected: {unique_classes}")
        return num_classes

    def get_dataloader(self):
        """ Returns a DataLoader with the balanced batch sampler. """
        return DataLoader(
            self.dataset,
            batch_sampler=self.batch_sampler,
            # collate_fn=self.custom_collate_fn,
        )
    
class PrepareData:
    """Prepare datasets in processing the audio samples from train, val, test dirs."""
    def __init__(self, args, csv_file_path, seg_len):
        self.args = args
        self.csv = csv_file_path
        self.seg_len = seg_len
        self.device = args.device

    def prepare(self):
        num_classes = DatasetValidator.get_num_classes_from_csv(self.csv)
        classnames = DatasetValidator.get_num_classes_from_csv(self.csv)
        train_dataset = ProcessDataset('train', self.csv, self.args, self.seg_len)
        test_dataset = ProcessDataset('test', self.csv, self.args, self.seg_len)
        val_dataset = ProcessDataset('val', self.csv, self.args, self.seg_len)

        train_loader = BalancedDataLoader(train_dataset.get_data(), self.args).get_dataloader()
        test_loader = DataLoader(test_dataset.get_data(), batch_size=64)
        val_loader = DataLoader(val_dataset.get_data(), batch_size=64)
        
        print('Data successfully loaded into DataLoaders.')

        return train_loader, test_loader, val_loader, num_classes, classnames

class SaveResultsToTensorboard:
    @staticmethod
    def get_class_names(csv_file_path):
        data = pd.read_csv(csv_file_path)
        data = data[data['set'] == 'train']
        label_map = {label: idx for idx, label in enumerate(sorted(data['label'].unique()))}

        return label_map

    @staticmethod
    def upload(stacked_metrics, cm, csv_file_path, writerTensorboard):
        """Upload the results to tensorboard."""
        label_map = SaveResultsToTensorboard.get_class_names(csv_file_path)
        labels = sorted(label_map.keys())

        acc, pre, rec, f1, loss = stacked_metrics.tolist()

        dictResult = {'Accuracy': '%.4f'%acc, 'Precision': '%.4f'%pre, 'Macro F1 Score': '%.4f'%f1, 'Recall': '%.4f'%rec, 'Loss': '%.8f'%loss}

        writerTensorboard.add_text('Results', Dict2MDTable.apply(dictResult), 0)

        cm_np = cm.cpu().numpy()
        df_cm = pd.DataFrame(cm_np, index=labels, columns=labels)

        df_cm_normalized = df_cm.div(df_cm.sum(axis=1), axis=0) * 100

        plt.figure(figsize=(12,7))
        plt.xticks(rotation=45)
        cm_heatmap = sn.heatmap(df_cm_normalized, annot=True, fmt=".2f", cmap="Blues").get_figure()

        writerTensorboard.add_figure('Confusion Matrix', cm_heatmap, 0)
        writerTensorboard.close()
        
class SaveResultsToDisk:
    @staticmethod
    def get_class_names(csv_file_path):
        """Retrives class names from dataset_split.csv file."""
        data = pd.read_csv(csv_file_path)
        data = data[data['set'] == 'train']
        label_map = {label: idx for idx, label in enumerate(sorted(data['label'].unique()))}
        return label_map

    @staticmethod
    def save_to_disk(args, stacked_metrics, cm, csv_file_path, current_run):
        """Save the results to disk as a CSV file."""
        label_map = SaveResultsToDisk.get_class_names(csv_file_path)
        labels = sorted(label_map.keys())

        log_dir = os.path.join(os.getcwd(), 'logs')
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        current_log_dir = os.path.join(log_dir, os.path.basename(current_run))
        if not os.path.exists(current_log_dir):
            os.makedirs(current_log_dir) 

        csv_path = os.path.join(current_log_dir, f'results_{os.path.basename(current_run)}.csv')

        acc, pre, rec, f1, loss = stacked_metrics.tolist()
        accuracy = '%.4f'%acc
        precision = '%.4f'%pre
        f1 = '%.4f'%f1
        recall = '%.4f'%rec
        loss = '%.8f'%loss

        write_header = not os.path.exists(csv_path)

        with open(csv_path, mode='a', newline='') as csv_file:
            writer = csv.writer(csv_file)

            if write_header:
                writer.writerow([
                    'Run Name', 'Model Name', 'Sample Rate', 'Segment Overlap',
                    'Fmin', 'Learning Rate', 'Epochs', 'Offline Augmentations', 'Online Augmentations', 'Early Stopping',
                    'Reduce LR on Plateau', 'Reduce LR by steps', 'Padding', 'Accuracy', 'Precision', 'Recall', 'F1 Score', 'Loss'
                ])

            writer.writerow([
                args.name, args.config, args.sr, args.segment_overlap,
                args.fmin, args.lr, args.epochs, args.offline_augment, args.online_augment,
                args.early_stopping, args.reduce_lr,  args.step_lr, args.padding,
                accuracy, precision, recall, f1, loss
            ])  

        cm_path = os.path.join(current_log_dir, f'cm_{os.path.basename(current_run)}.csv')
        cm_np = cm.cpu().numpy()
        df_cm = pd.DataFrame(cm_np, index=labels, columns=labels)
        df_cm.to_csv(cm_path)
        
class SaveYAML:
    """Save Hyperparameters to disk as YAML file."""
    @staticmethod
    def save_to_disk(args, num_classes, current_run):
        cwd = os.getcwd()
        path_to_run = current_run

        if not os.path.exists(os.path.join(cwd, 'runs')):
            os.makedirs(os.path.join(cwd, 'runs'))

        if not os.path.exists(path_to_run):
            os.makedirs(path_to_run)

        current_config = {'Name':args.name, 'Model': args.config , 'Sampling Rate':args.sr,'Segment Overlap':args.segment_overlap,
                      'Fmin':args.fmin, 'Learning Rate': args.lr, 'Epochs': args.epochs, 'Offline Augmentations':args.offline_augment,
                      'Online Augmentations': args.online_augment.split(), 'Early Stopping':args.early_stopping,'Reduce LR on Plateau':args.reduce_lr, 'Step LR': args.step_lr,
                      'Number of Classes':num_classes}

        yaml_file = os.path.join(path_to_run, f'{os.path.basename(path_to_run)}.yaml')

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

class Dict2MDTable:
    @staticmethod
    def apply(d, key='Name', val='Value'):
        """Convert args dictionnary to markdown table"""
        rows = [f'| {key} | {val} |']
        rows += ['|--|--|']
        rows += [f'| {k} | {v} |' for k, v in d.items()]
        return "  \n".join(rows)