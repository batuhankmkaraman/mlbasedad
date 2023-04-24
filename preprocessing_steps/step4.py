import pandas as pd
import numpy as np
import os
import random

class Step4:
    def __init__(self):
        """
        Object that reads the outputs of Step3 and generates random train/val/test splits.
        """  
        
    def read(self, path_to_read):
        """
        Read the outputs of Step3. The operations will be applied to self.df.
        
        Args:
            path_to_read (str): The directory containing the ADNIMERGE_DICT.csv and ADNIMERGE.csv files.  
        """
        path_to_read = path_to_read
        self.df_raw = pd.read_csv(path_to_read+'preprocessing_outputs/step3.csv', low_memory=False)
        self.df_dict_raw = pd.read_csv(path_to_read+'preprocessing_outputs/step3_DICT.csv', low_memory=False)
        self.df = self.df_raw.copy()
        self.df_dict = self.df_dict_raw.copy()
        
    def generate_stratified_splits(self, r_val, r_test, N_RT, N_RV, path_to_write, seed):
        """
        Generate stratified train/val/test splits.
        
        Args:
            r_val (between 0 and 1): Validation data ratio.
            r_test (between 0 and 1): Test data ratio.
            N_RT (int): Number of random experiments (trainval/test) splits.
            N_RV (int): Number of train/val splits for each trainval/test split.
            path_to_write : _description_
            seed (int): Randomness seed is set to this number.
        """
        # Set the seeds for a reproducible splitting.
        def set_random_seed(seed):
            random.seed(seed)
            np.random.seed(seed)
        set_random_seed(seed)
        # Check if path_to_write exists.
        if not os.path.exists(path_to_write):
            os.makedirs(path_to_write)
        # Main operation
        def operation(df, df_dict, r_val, r_test, N_RT, N_RV, path_to_write):
            r_val_relative = r_val/(1-r_test)
            for i_rt in range(N_RT):
                df_rvs = [df.copy() for z in range(N_RV)]
                for traj_label in np.unique(df['TRAJ_LABEL']):
                    # Get the indices corresponding to the current trajectory label.
                    traj_indices = np.array(df.loc[df['TRAJ_LABEL']==traj_label].index)
                    # Shuffle the indices and determine trainval/test splits.
                    np.random.shuffle(traj_indices)
                    N = traj_indices.size
                    indices_trainval, indicer_test = traj_indices[:int(N*(1-r_test))], traj_indices[int(N*(1-r_test)):]
                    for i_rv in range(N_RV):
                        # Shuffle the indices and determine train/val splits.
                        np.random.shuffle(indices_trainval)
                        N = indices_trainval.size
                        indices_train, indicer_val = indices_trainval[:int(N*(1-r_val_relative))], indices_trainval[int(N*(1-r_val_relative)):]
                        # Write down to the 'TrainValTest' column.
                        df_rvs[i_rv].loc[indices_train, 'TrainValTest'] = 'Train'
                        df_rvs[i_rv].loc[indicer_val, 'TrainValTest'] = 'Val'
                        df_rvs[i_rv].loc[indicer_test, 'TrainValTest'] = 'Test'
                # Write the generated random splits.
                rt_dir = path_to_write+'splits/'+'rt'+str(i_rt)+'/'
                if not os.path.exists(rt_dir):
                    os.makedirs(rt_dir)
                for i_rv in range(N_RV):
                    rv_dir = rt_dir + 'rv'+str(i_rv)+'/'
                    if not os.path.exists(rv_dir):
                        os.makedirs(rv_dir)
                    df_rvs[i_rv].to_csv(rv_dir + '/random_dataset.csv')
            # Update the dictionary and write.
            random_dataset_DICT = df_dict.copy()
            random_dataset_DICT.loc[len(random_dataset_DICT)] = ['TrainValTest', 'Other', 'Cat', 
                                                            'Indicates whether the subject is in train, validation, or test set.', pd.NA]
            random_dataset_DICT.to_csv(path_to_write+'dataset_DICT_for_splits.csv')
            
        operation(self.df, self.df_dict, r_val, r_test, N_RT, N_RV, path_to_write)
        
    def apply_all(self, args):
        """
        Apply the methods. 
        """
        self.read(args.path_to_read)
        self.generate_stratified_splits(r_test=args.r_test, r_val=args.r_val, N_RT=args.N_RT, N_RV=args.N_RV, 
                                        path_to_write=args.path_to_read, seed=args.seed)
        print('Step 4 is complete.')
   