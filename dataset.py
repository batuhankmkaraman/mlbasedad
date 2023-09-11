import torch
import pandas as pd
import numpy as np

class ADNI_tabular(torch.utils.data.Dataset):
    def __init__(self, trainvaltest: str, is_multi: bool, fixed_followup_year: int = 0,
                 synthetic_missingness_feats: list = None, is_progression: bool = False, data_dir: str = './'):
        """
        Constructor for the ADNI_tabular class.

        Args:
            trainvaltest (str): A string indicating the split of interest ('train', 'val', or 'test').
            is_multi (bool): A flag indicating whether to include all follow-up years (True) or only a specific year (False).
            fixed_followup_year (int, optional): The specific year of follow-up to include (ignored if is_multi is True). Defaults to 0.
            synthetic_missingness_feats (list, optional): A list of strings indicating the features to artificially introduce missingness into. If None, no features will be artificially made missing. Defaults to None.
            is_progression (bool, optional): A flag indicating whether to predict disease progression risk. If True, a new sample point is returned for each month between years 1 and 5. Defaults to False.
            data_dir (str, optional): A string indicating the directory where the dataset files are located. Defaults to './'.
        """
        
        # Read the dataset files.
        dataset_path = data_dir
        df = pd.read_csv(dataset_path+'random_dataset.csv', low_memory=False)
        sample_points = pd.read_csv(dataset_path+'sample_points.csv', low_memory=False)
        missingness_mask = pd.read_csv(dataset_path+'missingness_mask.csv', low_memory=False)
        preprocessed_features = pd.read_csv(dataset_path+'preprocessed_features.csv', low_memory=False)
        sample_weights = pd.read_csv(dataset_path+'sample_weights.csv', low_memory=False)
        
        # Get the data of split of interest.
        split_idx = sample_points['TrainValTest'] == trainvaltest
        sample_points = sample_points.loc[split_idx].reset_index(drop=True)
        missingness_mask = missingness_mask.loc[split_idx].reset_index(drop=True)
        preprocessed_features = preprocessed_features.loc[split_idx].reset_index(drop=True)
        sample_weights = sample_weights.loc[split_idx].reset_index(drop=True)
        
        # Get the data of follow-up year.
        if is_multi:
            years = [1,2,3,4,5]  
        else:
            years = [fixed_followup_year]
        year_idx = sample_points['FollowupYear'].isin(years)
        sample_points = sample_points.loc[year_idx].reset_index(drop=True)
        missingness_mask = missingness_mask.loc[year_idx].reset_index(drop=True)
        preprocessed_features = preprocessed_features.loc[year_idx].reset_index(drop=True)
        sample_weights = sample_weights.loc[year_idx].reset_index(drop=True)   

        # Scale sample weights.
        sample_weights *= len(sample_weights) / np.sum(sample_weights)    
        
        # If synthetic missingness is required for inference, it is created below.
        if synthetic_missingness_feats is not None:
            preprocessed_features[synthetic_missingness_feats] = 0  
            missingness_mask[['MASK_'+z for z in synthetic_missingness_feats]] = 1  
            
        # If disease progression risk predictions are required for inference, a new sample point is returned for each month between years 1 and 5.
        if is_progression:
            # Get the first followup visit of each subject.
            idx = sample_points.copy().drop_duplicates(subset=['RID'], keep='first').index
            new_sample_points = sample_points.loc[idx].reset_index(drop=True)
            new_missingness_mask = missingness_mask.loc[idx].reset_index(drop=True)
            new_preprocessed_features = preprocessed_features.loc[idx].reset_index(drop=True)
            new_sample_weights = sample_weights.loc[idx].reset_index(drop=True)
            # Repeat each row 49 times to sweep delta_t from 12 to 60.
            num_repetitions = 49
            num_rows = len(new_sample_points)
            new_sample_points = pd.concat([new_sample_points]*num_repetitions, ignore_index=True)
            new_missingness_mask = pd.concat([new_missingness_mask]*num_repetitions, ignore_index=True)
            new_preprocessed_features = pd.concat([new_preprocessed_features]*num_repetitions, ignore_index=True)
            new_sample_weights = pd.concat([new_sample_weights]*num_repetitions, ignore_index=True)
            # Write the new delta_t values.
            new_sample_points['delta_t'] = np.repeat(range(12, 61), num_rows)
            # Save.
            sample_points = new_sample_points.copy()
            missingness_mask = new_missingness_mask.copy()
            preprocessed_features = new_preprocessed_features.copy()
            sample_weights = new_sample_weights.copy()
                     
        # Save the relevant data to instance variables.
        self.df = df
        self.sample_idx = sample_points.index.values
        self.sample_points = sample_points
        self.missingness_mask = missingness_mask
        self.preprocessed_features = preprocessed_features
        self.sample_weights = sample_weights
        self.df_dict = {'CN':0, 'MCI':1, 'Dementia':2}  # A dictionary mapping diagnostic categories to numerical values.
        
    def __getitem__(self, index):
        """
        Method to get the data item at a given index.

        Parameters:
        -----------
        index: int
            Index of the data item to get.

        Returns:
        --------
        dict:
            A dictionary containing the data item at the given index.
        """
        batch = {}

        # Get the data items for the current index and add them to the batch dictionary.
        batch['SampleIndex'] = self.sample_idx[index]
        batch['RID'] = self.sample_points.loc[index, 'RID']
        batch['BaselineDX'] = self.df_dict[self.sample_points.loc[index, 'BaselineDX']]
        batch['FollowupYear'] = self.sample_points.loc[index, 'FollowupYear']
        batch['delta_t'] = self.sample_points.loc[index, 'delta_t']
        batch['FollowupDX'] = self.df_dict[self.sample_points.loc[index, 'FollowupDX']]
        batch['Features'] = self.preprocessed_features.loc[index].values
        batch['MissingnessMask'] = self.missingness_mask.loc[index].values
        batch['SampleWeight'] = self.sample_weights.loc[index, 'SampleWeight']

        return batch
    
    def __len__(self):
        """
        Method to get the length of the dataset.

        Returns:
        --------
        int:
            The length of the dataset.
        """
        return len(self.sample_points)

