# Args
import argparse
# Torch
import torch
import torch.nn.functional as F
# Fundamentals
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
from utils.metrics import get_roc_aucs, get_roc_curves
from utils.directory_navigator import DirectoryNavigator
from train import run_epoch

def get_ensembled_preds(i_rt, loader, network, dir_navi, args):
    """
    Perform ensembling of predictions for a given input configuration.

    Args:
    - i_rt (int): the value of i_rt for the current configuration.
    - loader (torch.utils.data.DataLoader): the data loader for the input dataset.
    - network (torch.nn.Module): the neural network model to use for ensembling.
    - dir_navi (DirectoryNavigator): the directory navigator object used for navigating directories.
    - args (argparse.Namespace): the argument namespace object.

    Returns:
    - pandas.DataFrame: the ensemble predictions as a pandas DataFrame.
    """
    # Initialize ensemble variables.
    ens_df = pd.DataFrame()
    ens_denom = 0.
    
    # Loop over all values of i_rv and i_ri for the current i_rt.
    for i_rv in range(args.N_RV):
        for i_ri in range(args.N_RI):
            # Load the trained model for the current configuration.
            network.load_state_dict(torch.load(dir_navi.get_training_results_dir(i_rt, i_rv, i_ri)+'model_weights.pt', map_location=args.device))
            
            # Make predictions for the current configuration.
            loss, df = run_epoch(loader, network, None, torch.nn.NLLLoss(reduction='none'), False, args)
            
            # If this is the first configuration, initialize the ensemble DataFrame.
            if len(ens_df) == 0:
                ens_df = df.copy()
            else:
                # Add the predictions to the ensemble DataFrame.
                ens_df['Pred'] += df['Pred']
            
            # Increment the denominator for the ensemble normalization.
            ens_denom += 1
    
    # Normalize the predictions by dividing by the number of configurations.
    ens_df['Pred'] /= ens_denom
    
    return ens_df

def get_eval_1_roc_auc_results(i_rt, network, dir_navi, args):
    """
    Compute ROC AUC scores for a given trained model on the test set.

    Args:
        i_rt (int): The index of the trained model to evaluate.
        network (torch.nn.Module): The PyTorch model to evaluate.
        dir_navi (DirNavigator): An object that helps with directory navigation.
        args (argparse.Namespace): Command-line arguments.

    Returns:
        A dictionary with ROC AUC scores for each class label.
    """
    # Create dataloader for test set.
    dataset = ADNI_tabular(trainvaltest='Test', is_multi=args.is_multi, fixed_followup_year=args.fixed_followup_year, data_dir=dir_navi.get_data_dir(i_rt, i_rv=0)) 
    loader =  torch.utils.data.DataLoader(dataset, batch_size=len(dataset), shuffle=False)
    
    # Get ensembled predictions for test set.
    ens_df = get_ensembled_preds(i_rt, loader, network, dir_navi, args)
    
    # Compute ROC AUC scores for each class label.
    roc_aucs = get_roc_aucs(ens_df)
    
    # Save predictions and metrics.
    dir_navi.create_eval_x_dir(1, i_rt)
    ens_df.to_csv(dir_navi.get_eval_x_dir(1, i_rt)+'test_preds.csv', index=False)
    pd.DataFrame([roc_aucs], index=[i_rt]).to_csv(dir_navi.get_eval_x_dir(1, i_rt)+'test_metrics.csv', index=False)
    
    return roc_aucs

        
def get_eval_2_feature_importance_results(i_rt, network, dir_navi, args): #TODO: return non-missing portion
    """
    Ensembles predictions for the test set and computes ROC AUC for different synthetic missingness scenarios.

    Args:
        i_rt (int): index of the current run.
        network (torch.nn.Module): trained PyTorch model to evaluate.
        dir_navi (Navigation): object to navigate through directories.
        args (Namespace): collection of command-line arguments.
    """
    
    # Get the feature dictionary for the synthetic missingness scenarios.
    df_dict = pd.read_csv(args.root+'dataset_DICT_for_splits.csv', low_memory=False)
    csf = df_dict.loc[df_dict['Modality']=='CSF', 'FLDNAME'].values.tolist()
    mri = df_dict.loc[df_dict['Modality']=='MRI', 'FLDNAME'].values.tolist()
    pet = df_dict.loc[df_dict['Modality']=='PET', 'FLDNAME'].values.tolist()
    feature_dict = {}
    feature_dict['cd'] = csf + mri + pet
    feature_dict['cd+av45'] = csf + mri + [z for z in pet if z != 'AV45']
    feature_dict['cd+csf'] = mri + pet
    feature_dict['cd+fdg'] = csf + mri + [z for z in pet if z != 'FDG']
    feature_dict['cd+hippo'] = csf + pet + [z for z in mri if z not in ['Hippocampus', 'ICV']]
    feature_dict['cd+mri'] = csf + pet
    feature_dict['cd+all'] = []
    
    # Evaluate for each synthetic missingness scenario.
    for feature_key, feature_list in feature_dict.items():
        # Create dataloader.
        dataset = ADNI_tabular(trainvaltest='Test', is_multi=args.is_multi, fixed_followup_year=args.fixed_followup_year, data_dir=dir_navi.get_data_dir(i_rt, i_rv=0), synthetic_missingness_feats=feature_list) 
        loader =  torch.utils.data.DataLoader(dataset, batch_size=len(dataset), shuffle=False)
        # Get ensembled predictions.
        ens_df = get_ensembled_preds(i_rt, loader, network, dir_navi, args)
        # Get ROC AUCs.
        roc_aucs = get_roc_aucs(ens_df)
        # Save predictions and metrics.
        dir_navi.create_eval_x_dir(2, i_rt)
        ens_df.to_csv(dir_navi.get_eval_x_dir(2, i_rt)+'test_preds_'+feature_key+'.csv', index=False)
        pd.DataFrame([roc_aucs], index=[i_rt]).to_csv(dir_navi.get_eval_x_dir(2, i_rt)+'test_metrics_'+feature_key+'.csv', index=False)

            
def get_eval_3_roc_curves_results(i_rt, network, dir_navi, args):
    """
    Computes and saves ROC curves for the test set.

    Args:
        i_rt (int): index of the target risk type
        network (nn.Module): neural network to use for prediction
        dir_navi (DirectoryNavigator): directory navigator object for handling file I/O
        args (argparse.Namespace): command line arguments

    Returns:
        None
    """
    # Create dataloader.
    dataset = ADNI_tabular(trainvaltest='Test', is_multi=args.is_multi, fixed_followup_year=args.fixed_followup_year, data_dir=dir_navi.get_data_dir(i_rt, i_rv=0)) 
    loader =  torch.utils.data.DataLoader(dataset, batch_size=len(dataset), shuffle=False)
    # Get ensembled predictions.
    ens_df = get_ensembled_preds(i_rt, loader, network, dir_navi, args)
    # Get roc curves.
    roc_curves = pd.DataFrame.from_dict(get_roc_curves(ens_df), orient='index').T
    # Save roc curves.
    dir_navi.create_eval_x_dir(3, i_rt)
    roc_curves.to_csv(dir_navi.get_eval_x_dir(3, i_rt)+'test_roc_curves.csv', index=False)

    
def get_eval_4_progression_results(i_rt, network, dir_navi, args):
    """
    Ensembles predictions for the test set with is_progression=True and saves results.

    Args:
        i_rt (int): Index of the run.
        network (_type_): PyTorch network to use.
        dir_navi (_type_): DirectoryNavigator object.
        args (_type_): Namespace object containing the script's arguments.
    """
    
    # Create dataloader.
    dataset = ADNI_tabular(trainvaltest='Test', is_multi=args.is_multi, fixed_followup_year=args.fixed_followup_year, data_dir=dir_navi.get_data_dir(i_rt, i_rv=0), is_progression=True) 
    loader = torch.utils.data.DataLoader(dataset, batch_size=len(dataset), shuffle=False)
    
    # Get ensembled predictions.
    ens_df = get_ensembled_preds(i_rt, loader, network, dir_navi, args)
    
    # Edit columns to avoid misconceptions.
    ens_df['FollowupYear'] = (ens_df['delta_t'] / 12.0).round(2)
    ens_df['FollowupDX'] = 'N/A'
    ens_df['SampleWeight'] = 'N/A' 
    
    # Save predictions.
    dir_navi.create_eval_x_dir(4, i_rt)
    ens_df.to_csv(dir_navi.get_eval_x_dir(4, i_rt) + 'test_preds.csv', index=False)

        
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
        - model_name (str): Name of the model.
        - model_width (int): The model width.
        - model_depth (int): The model depth.
        - num_classes (int): Number of classes in the classification task.
        - dropout_rate (float): Dropout rate for regularization.
        - batch_size (int): Batch size for training.
        - seed (int): Seed for random number generation.
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

        # Machine learning parameters
        self.add_argument('--model_name', type=str, default='NMM',
                          help='Name of the model to train')
        self.add_argument('--model_width', type=int, default=128,
                          help='Number of hidden units in each layer')
        self.add_argument('--model_depth', type=int, default=3,
                          help='Number of hidden layers')
        self.add_argument('--num_classes', type=int, default=3,
                          help='Number of classes')
        self.add_argument('--dropout_rate', type=float, default=0,
                          help='Dropout rate')

        self.add_argument('--batch_size', type=int, default=32,
                          help='Batch size')
        self.add_argument('--seed', type=int, default=1337,
                          help='Random seed')
        
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
    The main function that tests the machine learning models.

    Parses the command line arguments using the `Parser` class and creates a `DirectoryNavigator` object for
    navigating directories. Defines a dictionary of model names and corresponding models, creates the specified
    neural network model, and performs testing on the model.

    Returns:
    - None
    """

    # Parse arguments
    args = Parser().parse()

    # Create a directory navigator
    dir_navi = DirectoryNavigator(args.root, args.model_name, args.is_multi, args.fixed_followup_year)

    # Define a dictionary of model names and corresponding models
    models_dict = {
        'LSM_cross': LSM_cross,
        'LSM': LSM,
        'NSM': NSM,
        'NMM': NMM,
        'NMM_cross': NMM_cross,
        'NMM_3layer': NMM_3layer,
    }

    # Create model
    network = models_dict[args.model_name](
        input_size=args.num_of_tabular_features,
        model_width=args.model_width,
        model_depth=args.model_depth,
        dropout_rate=args.dropout_rate,
        num_classes=args.num_classes
    ).to(args.device)

    # Perform testing
    for i_rt in range(args.N_RT):
        """
        Loop over a range of values for `i_rt` and perform testing on the model.
        The testing consists of four different evaluations:
        1. ROC AUC results
        2. Feature importance results
        3. ROC curves results
        4. Progression results
        """
        print(f"Evaluating train/test split {i_rt} is in progress...")

        # Get evaluation 1: ROC AUC results
        get_eval_1_roc_auc_results(i_rt, network, dir_navi, args)

        # Get evaluation 2: Feature importance results
        get_eval_2_feature_importance_results(i_rt, network, dir_navi, args)

        # Get evaluation 3: ROC curves results
        get_eval_3_roc_curves_results(i_rt, network, dir_navi, args)

        # Get evaluation 4: Progression results
        get_eval_4_progression_results(i_rt, network, dir_navi, args)

        print(f"Evaluating train/test split {i_rt} is complete.")


if __name__ == '__main__':
    main()
