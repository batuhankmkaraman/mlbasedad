import pandas as pd
import numpy as np
import os

class Step5:
    def __init__(self):
        """
        Object that reads the outputs of Step4, performs feature preprocessing including imputation and normalization.
        """  
    
    def read_random_dataset(self, path_to_read):
        """
        Read a random dataset. The operations will be applied to self.df.
        
        Args:
            path_to_read (str): The directory containing the ADNIMERGE_DICT.csv and ADNIMERGE.csv files.  
        """
        self.df = pd.read_csv(path_to_read)
    
    def get_feature_names(self, df_dict, modalities_list=['Demographic', 'Genetic', 'PET', 'CSF', 'Clinical', 'Cognitive', 'MRI', 'Diagnosis']):
        feature_names = df_dict.loc[df_dict['Modality'].isin(modalities_list), 'FLDNAME'] 
        return feature_names 

    def get_sample_points(self, df, feature_names):
        sample_points = pd.DataFrame()
        for i in range(len(df)):
            for year in range(1, 6):
                if df.loc[i, 'FDX_'+str(year)] == df.loc[i, 'FDX_'+str(year)]:
                    #
                    sample_points.loc[len(sample_points), 'RID'] = df.loc[i, 'RID']
                    sample_points.loc[len(sample_points)-1, 'BaselineDX'] = df.loc[i, 'FDX_0']
                    sample_points.loc[len(sample_points)-1, 'FollowupYear'] = year
                    sample_points.loc[len(sample_points)-1, 'delta_t'] = df.loc[i, 'FMonth_'+str(year)]
                    sample_points.loc[len(sample_points)-1, 'FollowupDX'] = df.loc[i, 'FDX_'+str(year)]
                    sample_points.loc[len(sample_points)-1, 'TrainValTest'] = df.loc[i, 'TrainValTest']
                    sample_points.loc[len(sample_points)-1, 'IMAGEUID'] = df.loc[i, 'IMAGEUID']  
                    #
                    sample_points.loc[len(sample_points)-1, feature_names] = df.loc[i, feature_names]
        return sample_points
    
    def get_preprocessed_features(self, features, df_dict, train_idx):
        # Get the numerical and categorical feature columns.
        cat_cols = list(df_dict[df_dict['NumCat'] == 'Cat']['FLDNAME'])
        cat_cols = list(set(cat_cols).intersection(set(features.columns)))
        num_cols = list(set(features.columns) - set(cat_cols))
        # Perform imputation.
        train_modes_imp = features.loc[train_idx, cat_cols].mode()
        train_means_imp = features.loc[train_idx, num_cols].mean()
        features[cat_cols] = features[cat_cols].fillna(train_modes_imp)
        features[num_cols] = features[num_cols].fillna(train_means_imp)
        # Perform one-hot encoding.
        features = pd.get_dummies(features, columns=list(set(cat_cols)-set(['DX'])))
        features.loc[features['DX']=='CN', 'DX'] = 0.
        features.loc[features['DX']=='MCI', 'DX'] = 1.
        # Perform z-score normalization.
        train_means_norm = features[num_cols].loc[train_idx].mean()
        train_std_norm = features[num_cols].loc[train_idx].std()
        features[num_cols] = (features[num_cols] - train_means_norm) / train_std_norm
        return features
    
    def get_sample_weights(self, df):
        # Initialize an empty array to store the weights for each sample_point
        weights = np.zeros(df.shape[0])

        # Loop through each dataset split
        for split_name in ['Train', 'Val', 'Test']:
            split_df = df[df['TrainValTest'] == split_name]

            # Loop through each follow-up year
            for year in range(1, 6):

                # Get the subset of the dataframe for this follow-up year and dataset split
                year_split_df = split_df[split_df['FollowupYear'] == year]

                # Loop through each unique combination of BaselineDX and FollowupDX for this year and dataset split
                for (baseline_dx, followup_dx), count in year_split_df.groupby(['BaselineDX', 'FollowupDX']).size().items():

                    # Calculate the weight for this combination
                    weight = 1. / count

                    # Get the indices of the rows in the df that match this combination and dataset split
                    idxs = df.index[(df['TrainValTest'] == split_name) & (df['FollowupYear'] == year) & (df['BaselineDX'] == baseline_dx) & (df['FollowupDX'] == followup_dx)]

                    # Set the weight for each of these rows
                    weights[idxs] = weight
                
        # Scale the weights
        weights *= len(df) / np.sum(weights)

        # Write to a dataframe
        weights_df = pd.DataFrame()
        weights_df['SampleWeight'] = weights
        return weights_df

    def apply_all(self, args):
        """
        Apply the methods. 
        """
        print('Starting step 5...')
        
        df_dict = pd.read_csv(args.path_to_read+'dataset_DICT_for_splits.csv', low_memory=False)
        feature_names = self.get_feature_names(df_dict)
        
        for i_rt in range(args.N_RT):
            for i_rv in range(args.N_RV):
                path_to_split = args.path_to_read + 'splits/rt'+str(i_rt)+'/'+'rv'+str(i_rv)+'/'
                # Read the df.
                df = pd.read_csv(path_to_split+'random_dataset.csv', low_memory=False)
                # Apply the operations.
                sample_points = self.get_sample_points(df, feature_names)
                #
                train_idx = sample_points['TrainValTest'] == 'Train'
                features = sample_points[feature_names].copy()
                #
                missingness_mask = features.isnull().astype(int).add_prefix('MASK_')
                missingness_mask = missingness_mask.drop('MASK_DX', axis=1) # Drop the baseline DX mask. We know there's no missingness in baseline DX.
                #
                preprocessed_features = self.get_preprocessed_features(features, df_dict, train_idx)
                #
                sample_weights = self.get_sample_weights(sample_points)
                # Write.
                sample_points.to_csv(path_to_split+'sample_points.csv', index=False)
                missingness_mask.to_csv(path_to_split+'missingness_mask.csv', index=False)
                preprocessed_features.to_csv(path_to_split+'preprocessed_features.csv', index=False)
                sample_weights.to_csv(path_to_split+'sample_weights.csv', index=False)
                #
                print('Train/Val split', i_rv, 'of TrainVal/Test split', i_rt, 'is ready.')
        print('Step 5 is complete.')   
            
            

