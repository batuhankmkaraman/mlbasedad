# Args
import argparse
# Torch
import torch
import torch.nn.functional as F
# Fundamentals
import numpy as np
import random as random
import pandas as pd
# Datasets
from dataset import ADNI_tabular
# Models
from models.lsm_cross import LSM_cross
from models.lsm import LSM
from models.nsm import NSM
from models.nmm import NMM
from models.nmm_cross import NMM_cross
from models.nmm_3layer import NMM_3layer
# Utils
from utils.metrics import get_performance_metrics
from utils.directory_navigator import DirectoryNavigator

def train(network, optimizer, criterion, train_loader, val_loader, results_dir, args, tb_writer=None):
    """
    Trains a neural network using the given optimizer and loss function on a training set and
    monitors the performance on a validation set.

    Args:
        network (torch.nn.Module): The neural network to train.
        optimizer (torch.optim.Optimizer): The optimizer to use during training.
        criterion (torch.nn.Module): The loss function to use during training.
        train_loader (torch.utils.data.DataLoader): The data loader for the training set.
        val_loader (torch.utils.data.DataLoader): The data loader for the validation set.
        results_dir (str): The directory path to write the output files.
        args (argparse.Namespace): A configuration object with hyperparameters and other settings.
        tb_writer (torch.utils.tensorboard.SummaryWriter, optional): A TensorBoard SummaryWriter object for logging the training and validation losses and metrics.

    Returns:
        None

    Writes:
        - A pandas DataFrame that holds learning curves data, including the training and validation loss, balanced accuracy, and ROCAUC.
        - Additionally, please refer to the `process_val_loss()` function to learn about other written objects.
    """

    # Initialize variables.
    best_val_loss = float('inf')  # Initialize the best validation loss to infinity.
    patience = args.patience  # Set the patience to the specified value in the configuration.
    learning_curves = pd.DataFrame()  # Create an empty pandas DataFrame for storing the learning curves.

    # Main training loop.
    for epoch in range(1, args.max_epochs + 1):
        # Train for an epoch.
        train_loss, train_labels_preds = run_epoch(train_loader, network, optimizer, criterion, True, args)

        # Validation.
        val_loss, val_labels_preds = run_epoch(val_loader, network, None, criterion, False, args)

        # Get metrics for the training and validation sets.
        train_metrics = get_performance_metrics(train_labels_preds)  # Get the performance metrics for the training set.
        val_metrics = get_performance_metrics(val_labels_preds)  # Get the performance metrics for the validation set.
        train_metrics['Loss'] = train_loss  # Add the training loss to the training metrics dictionary.
        val_metrics['Loss'] = val_loss  # Add the validation loss to the validation metrics dictionary.

        # Save metrics.
        save_metrics(learning_curves, epoch, train_metrics, val_metrics)  # Update the learning curves DataFrame with the current epoch's metrics.
        if tb_writer is not None:
            write_to_tb(tb_writer, epoch, train_metrics, val_metrics)  # Write the current epoch's metrics to the TensorBoard log.

        # Check if the validation loss has improved.
        # If the validation loss hasn't improved for `patience` epochs, stop training.
        stop_training, best_val_loss, patience = process_val_loss(val_loss, best_val_loss, patience, results_dir, network, epoch, train_labels_preds, val_labels_preds, train_metrics, val_metrics, args)
        
        # If training should stop, break out of the loop.
        if stop_training:
            break

    # Write learning curves.
    learning_curves.to_csv(results_dir+'learning_curves.csv', index=False)  
    

def run_epoch(loader, network, optimizer, criterion, is_train, args):
    """
    Runs one epoch of training or evaluation on the specified data loader using the given network and optimizer.

    Args:
        loader (torch.utils.data.DataLoader): The PyTorch data loader containing the data to use for training or evaluation.
        network (torch.nn.Module): The PyTorch network to train or evaluate.
        optimizer (torch.optim.Optimizer): The PyTorch optimizer to use for training.
        criterion (torch.nn.Module): The PyTorch loss function to use for training or evaluation.
        is_train (bool): A flag indicating whether the network is being trained (True) or evaluated (False).
        args (argparse.Namespace): An object containing command-line arguments and hyperparameters.

    Returns:
        tuple: A tuple containing the average loss over the entire data set and a Pandas DataFrame containing the output
               probabilities and metadata for each sample in the data set.
    """
    
    # Initialize trackers for the running loss and epoch data frame
    running_loss = 0.0
    labels_preds_list = []

    # Set the network's mode to training or evaluation based on the is_train flag
    network.train(is_train)

    # Use PyTorch's context manager to automatically handle gradient computation
    with torch.set_grad_enabled(is_train):
        # Iterate over batches in the data loader
        for batch in loader:
            # Forward pass through the network to obtain the output logits
            if args.is_multi:
                # Use the appropriate forward pass for the multi-task model architecture
                output = network(batch['Features'].to(args.device), batch['MissingnessMask'].to(args.device), batch['delta_t'].to(args.device)) # This output is not softmaxed.
            else:
                # Use the appropriate forward pass for the single-task model architecture
                output = network(batch['Features'].to(args.device), batch['MissingnessMask'].to(args.device))

            # Compute the loss using the cross-entropy loss criterion with log-softmaxed output logits and the true labels
            loss_vector = criterion(torch.log_softmax(output, dim=1), batch['FollowupDX'].to(args.device))

            # Compute the weighted average of the loss across the batch
            loss = torch.mean(batch['SampleWeight'].to(args.device) * loss_vector)

            if is_train:
                # If in training mode, reset the optimizer's gradients to zero
                optimizer.zero_grad()
                # Backward pass to compute gradients and update network parameters
                loss.backward()
                optimizer.step()

            # Update running loss tracker
            running_loss += loss.detach().item() * len(batch['delta_t'])

            # Extract batch information and store in a temporary data frame
            meta_keys = ['SampleIndex', 'RID', 'BaselineDX', 'FollowupYear', 'delta_t', 'FollowupDX', 'SampleWeight']
            temp_df = pd.DataFrame({key: batch[key] for key in meta_keys})
            # Add the output probabilities for each class to the temporary data frame
            preds = F.softmax(output, dim=1).detach().cpu().numpy()
            temp_df['Pred'] = np.split(preds, preds.shape[0], axis=0)
            labels_preds_list.append(temp_df.copy())

    # Compute the average loss over the entire data set and concatenate the epoch data frame
    avg_loss = running_loss / len(loader.dataset)  
    labels_preds = pd.concat(labels_preds_list).sort_values(by='SampleIndex', ascending=True).reset_index(drop=True)

    # Return the average loss and the concatenated epoch data
    return avg_loss, labels_preds


def save_metrics(learning_curves, epoch, train_metrics, val_metrics):
    """
    Save the training and validation metrics for the current epoch to a pandas DataFrame.

    Args:
        learning_curves (pandas.DataFrame): A pandas DataFrame to save the metrics to.
        epoch (int): The current epoch number.
        train_metrics (dict): A dictionary of metric names and values for the training set.
        val_metrics (dict): A dictionary of metric names and values for the validation set.

    Returns:
        None
    """

    # Add a new row to the DataFrame for the current epoch.
    N = len(learning_curves)
    learning_curves.loc[N, 'Epoch'] = epoch
    
    # Save the training metrics to the DataFrame.
    for metric_key, metric_value in train_metrics.items():
        column_name = 'Train_' + metric_key
        learning_curves.loc[N, column_name] = metric_value
        
    # Save the validation metrics to the DataFrame.
    for metric_key, metric_value in val_metrics.items():
        column_name = 'Val_' + metric_key
        learning_curves.loc[N, column_name] = metric_value
        

def write_to_tb(tb_writer, epoch, train_metrics, val_metrics):
    """
    Writes the metrics to a tensorboard SummaryWriter object.

    Args:
        tb_writer (SummaryWriter): A tensorboard SummaryWriter object to write the metrics to.
        epoch (int): The current epoch number.
        train_metrics (dict): A dictionary of metric names and values for the training set.
        val_metrics (dict): A dictionary of metric names and values for the validation set.

    Returns:
        None
    """
    # Write train metrics.
    for metric_name, metric_value in train_metrics.items():   # Loop through each metric in the train_metrics dictionary.
        # Write the metric to the tensorboard writer for the train split.
        tb_writer.add_scalars(metric_name, {'Train': metric_value}, epoch)

    # Write validation metrics.
    for metric_name, metric_value in val_metrics.items():   # Loop through each metric in the val_metrics dictionary.
        # Write the metric to the tensorboard writer for the validation split.
        tb_writer.add_scalars(metric_name, {'Val': metric_value}, epoch)


def process_val_loss(val_loss, best_val_loss, patience, results_dir, network, epoch, train_labels_preds, val_labels_preds, train_metrics, val_metrics, args):
    """
    Process the validation loss for early stopping and write results to disk.

    Args:
    - val_loss (float): The validation loss for the current epoch.
    - best_val_loss (float): The best validation loss so far.
    - epoch (int): The current epoch number.
    - network (nn.Module): The neural network model.
    - train_labels_preds (pd.DataFrame): A pandas dataframe containing the predicted labels and actual labels for the training set.
    - val_labels_preds (pd.DataFrame): A pandas dataframe containing the predicted labels and actual labels for the validation set.
    - train_metrics (dict): A dictionary of metric names and values for the training set.
    - val_metrics (dict): A dictionary of metric names and values for the validation set.
    - args (Config): The configuration object.
    - patience (int): The number of epochs to wait before early stopping.
    - results_dir (str): The directory to write the results to.

    Returns:
    - stop_training (bool): A boolean indicating whether to stop training or not.
    - best_val_loss (float): The best validation loss so far.
    - patience (int): The number of epochs to wait before early stopping.
    """
    if val_loss < best_val_loss:
        # if the current validation loss is better than the best validation loss so far
        best_val_loss = val_loss
        # update the best validation loss
        torch.save(network.state_dict(), results_dir+'model_weights.pt')
        # save the current model weights
        
        # Write predicted labels and actual labels for the training and validation sets
        if args.write_preds:
            train_labels_preds.to_csv(results_dir+'train_labels_preds.csv', index=False)
            val_labels_preds.to_csv(results_dir+'val_labels_preds.csv', index=False)
        
        # Write the metric values for the training and validation sets
        if args.write_metrics:
            pd.DataFrame([train_metrics], index=[epoch]).to_csv(results_dir+'train_metrics.csv', index=False)
            pd.DataFrame([val_metrics], index=[epoch]).to_csv(results_dir+'val_metrics.csv', index=False)
        
        patience = args.patience
        # reset the patience counter
    else:
        patience -= 1
        # decrease the patience counter
        if patience == 0:
            print('Validation loss did not improve for {} epochs. Stopping training.'.format(args.patience))
            return True, best_val_loss, patience
            # early stopping
        
    return False, best_val_loss, patience
    # continue training
    

def set_seed(seed):
    """
    Set the random seed for different random number generators.

    Args:
    - seed (int): An integer value representing the random seed.

    Returns:
    - None

    This function sets the random seed for different random number generators used in Python libraries. This is useful for reproducibility of results in machine learning experiments.

    Random generators in different libraries generate random numbers based on the seed. Setting the same seed for different libraries ensures that they generate the same sequence of random numbers, which in turn makes the experiments reproducible.

    The function sets the random seed for the following libraries:
        - Python's built-in `random` module
        - NumPy library
        - PyTorch library for CPU and GPU

    Additionally, it ensures deterministic behavior when using CUDA by setting `torch.backends.cudnn.deterministic` to `True`.
    """
    random.seed(seed)   # set the random seed for the random module
    np.random.seed(seed)   # set the random seed for the numpy module
    torch.manual_seed(seed)   # set the random seed for the torch module for CPU
    torch.cuda.manual_seed(seed)   # set the random seed for the torch module for a single GPU
    torch.cuda.manual_seed_all(seed)   # set the random seed for the torch module for all available GPUs
    torch.backends.cudnn.deterministic = True   # ensure deterministic behavior when using CuDNN

        
class Parser(argparse.ArgumentParser):
    def __init__(self):
        """
        Custom argparse.ArgumentParser subclass for parsing command line arguments.

        Attributes:
        - fixed_followup_year (int): Fixed follow-up time for the single year models.
        - N_RT (int): Number of random train/test splits.
        - N_RV (int): Number of random train/val splis for each train/test split.
        - N_RI (int): Number of random initializations for each train/val split.
        - root (str): Root directory.
        - write_preds (bool): Whether to write predictions.
        - write_metrics (bool): Whether to write metrics.
        - write_tensorboard (bool): Whether to write to Tensorboard.
        - model_name (str): Name of the model.
        - model_width (int): The model width.
        - model_depth (int): The model depth.
        - num_classes (int): Number of classes in the classification task.
        - dropout_rate (float): Dropout rate for regularization.
        - batch_size (int): Batch size for training.
        - seed (int): Seed for random number generation.
        - max_epochs (int): Maximum number of epochs for training.
        - learning_rate (float): Learning rate for the optimizer.
        - weight_decay (float): Weight decay for regularization.
        - patience (int): Number of epochs to wait before early stopping.
        """
        
        super(Parser, self).__init__(description='parser')

        # I/O parameters
        self.add_argument('--fixed_followup_year', default=0,
                          type=int, help='Fixed followup year for single-year models.')
        
        self.add_argument('--N_RT', default=1,
                          type=int, help='Number of random train/test splits')
        self.add_argument('--N_RV', default=1,
                          type=int, help='Number of random train/val splis for each trainval/test split')
        self.add_argument('--N_RI', default=1,
                          type=int, help='Number of random initializations (of the model weights) for each train/val split')

        self.add_argument('--root', default='./',
                          type=str, help='Root directory for data and results')

        self.add_bool_arg('write_preds', True)
        self.add_bool_arg('write_metrics', True)
        self.add_bool_arg('write_tensorboard', False)

        # Machine learning parameters
        self.add_argument('--model_name', type=str, default='NMM',
                          help='Name of the model to train')
        self.add_argument('--model_width', type=int, default=128,
                          help='Number of hidden units in each layer')
        self.add_argument('--model_depth', type=int, default=3,
                          help='Number of hidden layers')
        self.add_argument('--num_classes', type=int, default=3,
                          help='Number of classes')
        self.add_argument('--dropout_rate', type=float, default=0.2,
                          help='Dropout rate')

        self.add_argument('--batch_size', type=int, default=32,
                          help='Batch size')
        self.add_argument('--seed', type=int, default=1337,
                          help='Random seed')
        self.add_argument('--max_epochs', type=int, default=10000,
                          help='Maximum number of epochs')
        self.add_argument('--learning_rate', type=float, default=1e-4,
                          help='Learning rate')
        self.add_argument('--weight_decay', type=float, default=1e-6,
                          help='Weight decay')
        self.add_argument('--patience', type=int, default=75,
                          help='Patience for early stopping')

    def add_bool_arg(self, name, default=False):
        """
        Add boolean argument to argparse parser.

        Args:
        - self: the current object instance.
        - name (str): the name of the argument to be added.
        - default (bool, optional): the default value of the argument. Defaults to False.

        Returns:
        - None
        """
        group = self.add_mutually_exclusive_group(required=False)
        group.add_argument('--' + name, dest=name, action='store_true')
        group.add_argument('--no_' + name, dest=name, action='store_false')
        self.set_defaults(**{name: default})
    
    def is_multi(self, args):
        """
        Add a single-year/multi-year training indicator to args.

        Args:
        - self: the current object instance.
        - args (argparse.Namespace): the argument namespace object.

        Returns:
        - argparse.Namespace: the updated argument namespace object.
        """
        if args.model_name in ['NMM', 'NMM_cross', 'NMM_3layer', 'NMM_optimized']:
            args.is_multi = True
        elif args.model_name in ['LSM', 'LSM_cross', 'NSM', 'NSM_CNN']:
            args.is_multi = False
        return args
        
    def get_device(self, args):
        """
        Get device and assign it to args.device.

        Args:
        - self: the current object instance.
        - args (argparse.Namespace): the argument namespace object.

        Returns:
        - argparse.Namespace: the updated argument namespace object.
        """
        args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
        return args
        
    def get_num_of_tabular_features(self, args):
        """
        Get the number of tabular features based on whether the model is single-year or multi-year.

        Args:
        - self: the current object instance.
        - args (argparse.Namespace): the argument namespace object.

        Returns:
        - argparse.Namespace: the updated argument namespace object.
        """
        if args.is_multi:
            args.num_of_tabular_features = 114
        else:
            args.num_of_tabular_features = 113
        return args

    def parse(self):
        """
        Parse the command-line arguments and print them.

        Args:
        - self: the current object instance.

        Returns:
        - argparse.Namespace: the argument namespace object.
        """
        
        # Parse command-line arguments
        args = self.parse_args()
        
        # Add multi-year training indicator, get device, and get the number of tabular features based on parsed arguments
        args = self.is_multi(args)
        args = self.get_device(args)
        args = self.get_num_of_tabular_features(args) 
        
        # Only splits folder is required.
        args.root += 'splits/'
        
        # Print parsed arguments
        print('Arguments:')
        print(vars(args))
                
        return args


def main():
    """
    The main function that runs the deep learning model.

    Returns:
    - None
    """

    # Parse arguments
    args = Parser().parse()
    
    # Create a directory navigator.
    dir_navi = DirectoryNavigator(args.root, args.model_name, args.is_multi, args.fixed_followup_year) 

    # Define a dictionary of model names and corresponding models.
    models_dict = {'LSM_cross':LSM_cross,   
                    'LSM':LSM,
                    'NSM':NSM,
                    'NMM':NMM,
                    'NMM_cross':NMM_cross,
                    'NMM_3layer':NMM_3layer,
                    }
    
    for i_rt in range(args.N_RT):   # loop over a range of values for i_rt
        for i_rv in range(args.N_RV):   # loop over a range of values for i_rv
            for i_ri in range(args.N_RI):   # loop over a range of values for i_ri
                
                # Set the random seed.
                set_seed(args.seed+i_ri)
                
                # Create results dir.
                dir_navi.create_training_results_dir(i_rt, i_rv, i_ri)
                
                # Create dataloaders.
                train_loader = torch.utils.data.DataLoader(
                    ADNI_tabular(trainvaltest='Train', is_multi=args.is_multi, 
                                 fixed_followup_year=args.fixed_followup_year, 
                                 data_dir=dir_navi.get_data_dir(i_rt, i_rv)), 
                                 batch_size=args.batch_size, shuffle=True)   # create a dataloader for the training set
                
                val_set = ADNI_tabular(trainvaltest='Val', is_multi=args.is_multi, 
                                       fixed_followup_year=args.fixed_followup_year, 
                                       data_dir=dir_navi.get_data_dir(i_rt, i_rv))   # create a dataset for the validation set
                
                val_loader =  torch.utils.data.DataLoader(val_set, batch_size=len(val_set), shuffle=False)   # create a dataloader for the validation set
                
                # Create ML objects.
                network = models_dict[args.model_name](
                    input_size=args.num_of_tabular_features, model_width=args.model_width, 
                    model_depth=args.model_depth, dropout_rate=args.dropout_rate, 
                    num_classes=args.num_classes).to(args.device)   # create the neural network model
                
                optimizer = torch.optim.Adam(network.parameters(), lr=args.learning_rate, 
                                             weight_decay=args.weight_decay)   # create an optimizer object
                
                criterion = torch.nn.NLLLoss(reduction='none')   # create cce loss function
                
                # Create TensorBoard writer if preferred by the user.
                if args.write_tensorboard:
                    from torch.utils.tensorboard import SummaryWriter
                    tb_writer = SummaryWriter(log_dir=dir_navi.get_tb_dir(i_rt, i_rv, i_ri))   # create a TensorBoard writer object
                else:
                    tb_writer = None
                
                # Perform training.
                print(i_rt, i_rv, i_ri, 'is training.')   # print a message indicating that the current configuration is being trained.
                train(network, optimizer, criterion, train_loader, val_loader, 
                      dir_navi.get_training_results_dir(i_rt, i_rv, i_ri), args, tb_writer)   # train the neural network
                
                print(i_rt, i_rv, i_ri, 'is complete.')   # print a message indicating that the current configuration is complete.

if __name__ == '__main__':
    main()
            


