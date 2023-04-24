import torch
import torch.nn as nn
import torch.nn.functional as F

class LSM_cross(nn.Module):
    """
    Linear single year model that only has a single layer of width 3, i.e., a softmax (or multinominal) regression model.

    Args:
        input_size (int): The number of features in the input.
        num_classes (int): The number of classes to predict.

    Attributes:
        input_size (int): The number of features in the input.
        num_classes (int): The number of classes to predict.
        layers (nn.ModuleList): The list of fully connected layers.

    """

    def __init__(self, input_size=113, num_classes=3):
        super(LSM_cross, self).__init__()
        self.input_size = input_size
        self.num_classes = num_classes
        
        # Define fully connected layers
        self.layers = nn.ModuleList([nn.Linear(input_size, num_classes)])

    def forward(self, features, missingness_mask):
        """
        Forward pass of the neural network.

        Args:
            features (torch.Tensor): The input features of shape `(batch_size, input_size)`.
            missingness_mask (torch.Tensor): The binary mask indicating which features are missing of shape `(batch_size, input_size)`.

        Returns:
            torch.Tensor: The output logits of shape `(batch_size, num_classes)`.
        """
        x = torch.cat([features, missingness_mask], dim=1)
        x = x.float()
        
        x = self.layers(x)
        
        return x
