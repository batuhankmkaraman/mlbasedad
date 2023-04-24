import torch
import torch.nn as nn
import torch.nn.functional as F

from models.model_utils import get_layers

class NMM(nn.Module):
    """
    Nonlinear Multi-year Model.

    Args:
        input_size (int): The number of features in the input.
        model_width (int): The model width - Please check get_layers() from models.model_utils to see the actual use.
        model_depth (int): The model depth - Please check get_layers() from models.model_utils to see the actual use.
        dropout_rate (float): The probability of an element to be zeroed in dropout.
        num_classes (int): The number of classes to predict.

    Attributes:
        input_size (int): The number of features in the input.
        model_width (int): The number of neurons in each hidden layer.
        model_depth (int): The number of hidden layers.
        dropout_rate (float): The probability of an element to be zeroed in dropout.
        num_classes (int): The number of classes to predict.
        layers (nn.Sequential): The sequence of fully connected layers.

    """

    def __init__(self, input_size=114, model_width=128, model_depth=3, dropout_rate=0.2, num_classes=3):
        super(NMM, self).__init__()
        self.input_size = input_size
        self.model_width = model_width
        self.model_depth = model_depth
        self.dropout_rate = dropout_rate
        self.num_classes = num_classes
        
        # Get fully connected layers
        self.layers = get_layers(self.input_size, self.model_width, self.model_depth, self.num_classes)

    def forward(self, features, missingness_mask, delta_t):
        """
        Forward pass of the neural network.

        Args:
            features (torch.Tensor): The input features of shape `(batch_size, input_size)`.
            missingness_mask (torch.Tensor): The binary mask indicating which features are missing of shape `(batch_size, input_size)`.
            delta_t (torch.Tensor): The follow-up time.

        Returns:
            torch.Tensor: The output logits of shape `(batch_size, num_classes)`.
        """
        x = torch.cat([features, missingness_mask, delta_t.unsqueeze(dim=-1)], dim=1)
        x = x.float()

        for i, layer in enumerate(self.layers):
            if i != len(self.layers) - 1:
                x = F.relu(layer(x))
                x = F.dropout(x, p=self.dropout_rate, training=self.training)
            else:
                x = layer(x)
        
        return x
