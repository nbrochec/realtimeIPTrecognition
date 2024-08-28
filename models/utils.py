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

from models import v1, v2, one_residual, two_residual, transformer
import torch.nn.init as init

from tqdm import tqdm

from torchmetrics.classification import MulticlassAccuracy, MulticlassPrecision, MulticlassRecall, MulticlassF1Score

class LoadModel:
    def __init__(self):
        self.models = {
            'v1': v1,
            'v2': v2,
            'one_residual': one_residual,
            'two_residual': two_residual,
            'transformer': transformer,
        }
    
    def get_model(self, model_name, output_nbr):
        if model_name in self.models:
            return self.models[model_name](output_nbr)
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
        """
        Sum the number of parameters of the model.

        Returns
        -------
        int 
        """
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

        Returns
        -------
        torch.Tensor
            The output of the model for the random input tensor.
        """
        self.model.to(self.device)
        self.model.eval()  # Set the model to evaluation mode

        # Generate a random input tensor with the specified shape
        random_input = torch.randn(self.input_shape).to(self.device)
        
        # Forward pass through the model
        with torch.no_grad():
            output = self.model(random_input)
            # proba_distrib = F.softmax(output, dim=1)
        
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

    def initialize(self, init_method=None):
        """
        Apply weight initialization to the model layers.

        Parameters
        ----------
        init_method : callable, optional
            A custom initialization function. If None, Xavier-Glorot initialization is used.

        Returns
        -------
        torch.nn.Module
            The model with initialized weights.
        """
        if init_method is None:
            init_method = torch.nn.init.xavier_normal_

        for layer in self.model.modules():
            if isinstance(layer, (torch.nn.Conv2d, torch.nn.Linear)):
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

    def train_epoch(self, loader, optimizer, augmentations, aug_number):
        """
        Perform one training epoch.

        Parameters
        ----------
        loader : torch.utils.data.DataLoader
            The DataLoader for the training data.
        optimizer : torch.optim.Optimizer
            The optimizer used for training.
        augmentations : object
            An object responsible for data augmentations.
        aug_number : int
            The number of augmentations applied.

        Returns
        -------
        float
            The average loss for the epoch.
        """
        self.model.train()
        running_loss = 0.0

        for data, targets in tqdm(loader, desc="Training", leave=False):
            data, targets = data.to(self.device), targets.to(self.device)
            optimizer.zero_grad()

            # Apply augmentations
            augmented_data = augmentations.apply(data)
            new_data = torch.cat((data, augmented_data), dim=0)
            all_targets = torch.flatten(targets.repeat(aug_number + 1, 1))

            outputs = self.model(new_data)
            loss = self.loss_fn(outputs, all_targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * data.size(0)
        
        return running_loss / len(loader.dataset)

    def validate_epoch(self, loader):
        """
        Perform one validation epoch.

        Parameters
        ----------
        loader : torch.utils.data.DataLoader
            The DataLoader for the validation data.

        Returns
        -------
        float
            The average loss for the epoch.
        """
        self.model.eval()
        running_loss = 0.0
        with torch.no_grad():
            for data, targets in tqdm(loader, desc="Validation", leave=False):
                data, targets = data.to(self.device), targets.to(self.device)
                outputs = self.model(data)
                loss = self.loss_fn(outputs, targets)
                running_loss += loss.item() * data.size(0)
        
        return running_loss / len(loader.dataset)

    def test_model(self, loader):
        """
        Test the model and compute metrics.

        Parameters
        ----------
        loader : torch.utils.data.DataLoader
            The DataLoader for the test data.

        Returns
        -------
        tuple
            A tuple containing the average loss and accuracy.
        """
        self.model.eval()
        running_loss = 0.0
        total_samples = 0

        all_targets = []
        for data, targets in loader:
            all_targets.append(targets.cpu())
        all_targets = torch.cat(all_targets)
        class_nbr = len(torch.unique(all_targets))

        accuracy_metric = MulticlassAccuracy(num_classes=class_nbr).to(self.device)
        precision_metric = MulticlassPrecision(num_classes=class_nbr).to(self.device)
        recall_metric = MulticlassRecall(num_classes=class_nbr).to(self.device)
        f1_metric = MulticlassF1Score(num_classes=class_nbr, average='macro')

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

                running_loss += loss.item() * batch_size
                total_samples += batch_size

        accuracy = accuracy_metric.compute().item()
        precision = precision_metric.compute().item()
        recall = recall_metric.compute().item()
        f1 = f1_metric.compute().item()

        running_loss = running_loss / total_samples

        return accuracy, precision, recall, f1, running_loss