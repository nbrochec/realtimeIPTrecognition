#############################################################################
# utils.py 
# Nicolas Brochec
# TOKYO UNIVERSITY OF THE ARTS
# 東京藝術大学音楽音響創造科
# ERC Reach
# GNU General Public License v3.0
#############################################################################
# Code description:
# Implement utility functions
#############################################################################

import torch
import humanize
import sys
import pandas as pd

from models import v1, v2, v3, v2bis, v2_1d, v1_1d, v1_1d_e, v1_mi, v1_mi_1d, v1_mi4, v1_mi4_e, v1_mi4_hpss # one_residual, two_residual, transformer,
from models import v1_mi6
import torch.nn.init as init

from tqdm import tqdm

from torchmetrics.classification import MulticlassAccuracy, MulticlassPrecision, MulticlassRecall, MulticlassF1Score, MulticlassConfusionMatrix

class LoadModel:
    def __init__(self):
        self.models = {
            'v1': v1,
            'v1_1d': v1_1d,
            'v2': v2,
            'v3': v3,
            'v2bis': v2bis,
            # 'one-residual': one_residual,
            # 'two-residual': two_residual,
            # 'transformer': transformer,
            'v2_1d': v2_1d,
            'v1_1d_e': v1_1d_e,
            'v1_mi':v1_mi,
            'v1_mi_1d': v1_mi_1d,
            'v1_mi4': v1_mi4,
            'v1_mi4_e': v1_mi4_e,
            'v1_mi4_hpss': v1_mi4_hpss,
            'v1_mi6': v1_mi6
        }
    
    def get_model(self, model_name, output_nbr, sr):
        if model_name in self.models:
            return self.models[model_name](output_nbr, sr)
        else:
            raise ValueError(f"Model {model_name} is not recognized.")

class ModelSummary:
    def __init__(self, model, num_labels, config):
        """
        Initializes the ModelSummary class.

        Parameters
        ----------
        model : torch.nn.Module
            The model to be tested.
        num_labels : int
            The number of labels.
        config : str
            The chosen model's configuration.
        """
        self.model = model
        self.num_labels = num_labels
        self.config = config

    def get_total_parameters(self):
        return sum(p.numel() for p in self.model.parameters())

    def print_summary(self):
        total_params = self.get_total_parameters()
        formatted_params = humanize.intcomma(total_params)

        print('\n')
        print('-----------------------------------------------')
        print(f"Model Summary:")
        print(f"Model's name: {self.config}")
        print(f"Number of labels: {self.num_labels}")
        print(f"Total number of parameters: {formatted_params}")
        print('-----------------------------------------------')
        print('\n')

class ModelTester:
    def __init__(self, model, input_shape=(1, 1, 7680), device='cpu'):
        """
        Initializes the ModelTester class.

        Parameters
        ----------
        model : torch.nn.Module
            The model to be tested.
        input_shape : tuple
            The shape of the input data (default is (1, 1, 7680)).
        device : str
            The device to run the model on ('cpu' or 'cuda').
        """
        self.model = model
        self.input_shape = input_shape
        self.device = device

    def test(self):
        """
        Tests the model with a random input tensor.
        """
        self.model.to(self.device)
        self.model.eval()

        random_input = torch.randn(self.input_shape).to(self.device)
        
        with torch.no_grad():
            output = self.model(random_input)
        
        return output
    
class ModelInit:
    def __init__(self, model):
        """
        Initialize the ModelInit class.

        Parameters
        ----------
        model : torch.nn.Module
            The model whose weights need to be initialized.
        """
        self.model = model

    def initialize(self):
        """
        Apply weight initialization to the model layers.
        """
        init_method = torch.nn.init.xavier_normal_

        for layer in self.model.modules():
            if isinstance(layer, (torch.nn.Conv2d, torch.nn.Linear, torch.nn.Conv1d)):
                init_method(layer.weight)
                if layer.bias is not None:
                    torch.nn.init.zeros_(layer.bias)

        return self.model
    
class ModelTrainer:
    def __init__(self, model, loss_fn, device):
        """
        Initialize the ModelTrainer.

        Parameters
        ----------
        model : torch.nn.Module
            The model to be trained, validated, and tested.
        loss_fn : callable
            The loss function used for training and evaluation.
        device : torch.device
            The device to which the model and data will be moved.
        """
        self.model = model
        self.loss_fn = loss_fn
        self.device = device

    def train_epoch(self, loader, optimizer, augmenter):
        """
        Perform one training epoch.
        """
        self.model.train()
        running_loss = 0.0

        for data, targets in tqdm(loader, desc="Training", leave=False):
            data, targets = data.to(self.device), targets.to(self.device)
            optimizer.zero_grad()
            
            aug_data = augmenter(data)

            outputs = self.model(aug_data)
            loss = self.loss_fn(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * data.size(0)
        
        return running_loss / len(loader.dataset)

    def validate_epoch(self, loader):
        """
        Perform one validation epoch.
        """
        self.model.eval()
        running_loss = 0.0

        all_targets = []
        for data, targets in loader:
            all_targets.append(targets.cpu())
        all_targets = torch.cat(all_targets)
        class_nbr = len(torch.unique(all_targets))

        val_acc = MulticlassAccuracy(num_classes=class_nbr, average='micro').to(self.device)
        val_f1 = MulticlassF1Score(num_classes=class_nbr, average='macro').to(self.device)
        val_pre = MulticlassPrecision(num_classes=class_nbr).to(self.device).to(self.device)
        val_rec = MulticlassRecall(num_classes=class_nbr).to(self.device).to(self.device)

        with torch.no_grad():
            for data, targets in tqdm(loader, desc="Validation", leave=False):
                data, targets = data.to(self.device), targets.to(self.device)
                outputs = self.model(data)
                loss = self.loss_fn(outputs, targets)
                running_loss += loss.item() * data.size(0)
                _, predicted = torch.max(outputs, 1)

                val_acc.update(preds=predicted, target=targets)
                val_pre.update(preds=predicted, target=targets)
                val_rec.update(preds=predicted, target=targets)
                val_f1.update(preds=predicted, target=targets)

        val_loss = running_loss / len(loader.dataset)
        val_loss = torch.tensor(val_loss).to(self.device)
    
        val_acc = val_acc.compute().detach().cpu().item()
        val_pre = val_pre.compute().detach().cpu().item()
        val_rec = val_rec.compute().detach().cpu().item()
        val_f1 = val_f1.compute().detach().cpu().item()

        return val_loss, val_acc, val_pre, val_rec, val_f1

    def test_model(self, loader):
        """
        Test the model and compute metrics.
        """
        self.model.eval()
        running_loss = 0.0
        total_samples = 0

        all_targets = []
        for data, targets in loader:
            all_targets.append(targets.cpu())
        all_targets = torch.cat(all_targets)
        class_nbr = len(torch.unique(all_targets))

        accuracy_metric = MulticlassAccuracy(num_classes=class_nbr, average='micro').to(self.device)
        precision_metric = MulticlassPrecision(num_classes=class_nbr).to(self.device)
        recall_metric = MulticlassRecall(num_classes=class_nbr).to(self.device)
        f1_metric = MulticlassF1Score(num_classes=class_nbr, average='macro').to(self.device)
        cm_metric = MulticlassConfusionMatrix(num_classes=class_nbr).to(self.device)

        with torch.no_grad():
            for data, targets in tqdm(loader, desc="Test", leave=False):
                data, targets = data.to(self.device), targets.to(self.device)
                outputs = self.model(data)
                loss = self.loss_fn(outputs, targets)
                batch_size = data.size(0)
                _, predicted = torch.max(outputs, 1)

                accuracy_metric.update(preds=predicted, target=targets)
                precision_metric.update(preds=predicted, target=targets)
                recall_metric.update(preds=predicted, target=targets)
                f1_metric.update(preds=predicted, target=targets)
                cm_metric.update(preds=predicted, target=targets)

                running_loss += loss.item() * batch_size
                total_samples += batch_size

        accuracy = torch.round(accuracy_metric.compute(), decimals=6) * 100
        precision = torch.round(precision_metric.compute(), decimals=6) * 100
        recall = torch.round(recall_metric.compute(), decimals=6) * 100
        f1 = torch.round(f1_metric.compute(), decimals=6) * 100
        cm = cm_metric.compute()

        running_loss = running_loss / total_samples

        print(f'Test Loss: {running_loss:.4f}')
        print(f'Test Accuracy: {accuracy:.4f}')
        print(f'Test Precision: {precision:.4f}')
        print(f'Test Recall: {recall:.4f}')
        print(f'Test Macro F1 Score: {f1:.4f}')

        stacked_metrics = torch.stack([accuracy.to(self.device), precision.to(self.device), recall.to(self.device), f1.to(self.device), torch.tensor(running_loss).to(self.device)], dim=0)

        return stacked_metrics, cm
    
class PrepareModel:
    def __init__(self, args, num_classes, seg_len):
        self.args = args
        self.num_classes = num_classes
        self.seg_len = seg_len
        self.device = args.device

    def prepare(self):
        model = LoadModel().get_model(self.args.config, self.num_classes, self.args.sr).to(self.device)

        tester = ModelTester(model, input_shape=(1, 1, self.seg_len), device=self.device)
        output = tester.test()
        if output.size(1) != self.num_classes:
            print("Error: Output dimension does not match the number of classes.")
            sys.exit(1)
    
        summary = ModelSummary(model, self.num_classes, self.args.config)
        summary.print_summary()

        model = ModelInit(model).initialize()
        return model

        
