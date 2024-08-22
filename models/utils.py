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