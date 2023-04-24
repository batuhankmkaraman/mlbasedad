import torch.nn as nn

def get_layers(input_size, model_width, model_depth, num_classes):
    """
    Constructs a list of linear layers for a fully connected neural network.
    Note: "S1 Text. Details of NMM (3-layer) and NMM (optimized)" in the supplementary of the paper 
        explains the use of model_width and model_depth.
    
    Args:
        input_size (int): The number of input features.
        model_width (int): The number of hidden units in the first fully-connected block.
            Must be one of [64, 128, 256].
        model_depth (int): Number of fully-connected blocks. 
            Must be one of [1, 2, 3].
        num_classes (int): The number of output classes.

    Returns:
        nn.ModuleList: A list of linear layers.
    """
    assert model_width in [64, 128, 256], "Invalid width"
    assert model_depth in [1, 2, 3], "Invalid depth"
    
    layers = nn.ModuleList()
    
    # First model block
    layers.append(nn.Linear(input_size, model_width))
    last_input_width = model_width
    
    if model_depth >= 1:
        # Add 2 more linear layers
        layers.append(nn.Linear(model_width, model_width))
        layers.append(nn.Linear(model_width, model_width))
        
    if model_depth >= 2:
        # Add 5 more layers with width model_width//2
        layers.append(nn.Linear(model_width, model_width//2))
        layers.append(nn.Linear(model_width//2, model_width//2))
        layers.append(nn.Linear(model_width//2, model_width//2))
        layers.append(nn.Linear(model_width//2, model_width//2))
        layers.append(nn.Linear(model_width//2, model_width//2))
        last_input_width = model_width//2
        
    if model_depth >= 3:
        # Add 2 more layers with width//4
        layers.append(nn.Linear(model_width//2, model_width//4))
        layers.append(nn.Linear(model_width//4, model_width//4))
        last_input_width = model_width//4
        
    # Output layer
    layers.append(nn.Linear(last_input_width, num_classes))
    
    return layers
