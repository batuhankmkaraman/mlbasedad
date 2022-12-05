import pandas as pd
import random
import numpy as np
import pickle
import copy
import os

def set_random_seed(seed):
    """
    Assign a seed number for random generators to obtain reproducible results.
    """
    random.seed(seed)
    np.random.seed(seed)
set_random_seed(1337) 

class StratifiedDataSplitting:
    def __init__(self, directory_to_read, directory_to_save):
        """
        Reads selected participants' data and its dictionary, splits subjects' indices into train/val/test sets in a stratified fashion based on disease trajectories, updates the data and its dictionary by adding a new column called 'Cluster' that is used for stratification, and saves the results.
        Args:
            directory_to_read : Directory of selected participants' data.
            directory_to_save : Directory to save the results.
        Returns:
            Saves the updated data of selected participants as df_sds.csv.
            Saves the updated data dictionary as df_cols_sds.csv.
            Saves the indices of train/val/test splits as splits/tvt_x_y.txt where x and y correspond to train/test and train/val split identifiers, respectively. Recall that we have 200 train/test splits, and for each split, 5 train/val splits in the paper. Thus, x ranges from 0 to 199, and y ranges from 0 to 4.
        """
        self.directory_to_read = directory_to_read
        self.directory_to_save = directory_to_save
    
        def read():
            """
            Read selected participants' data.
            """
            self.df = pd.read_csv(self.directory_to_read+'df_ps.csv', low_memory=False)
            self.df_cols = pd.read_csv(self.directory_to_read+'df_cols_ps.csv', low_memory=False)

        def add_Cluster_column():
            """
            Cluster participants based on their disease trajectories. Note that the trajectory label columns (['TL_'+str(year) for year in range(1,6)]) are created in ParticipantSelection.py.
            """
            def _get_enumerated(trajs):
                """
                Helper function that maps the diagnosis to distinct integers.
                """
                trajs = trajs.fillna('Unknown')
                trajs = trajs.replace({'CN':0, 'MCI':1, 'Dementia':2, 'Unknown':-4})
                return trajs    
            # Get the disease trajectories as a numerical matrix.
            tl_cols = ['TL_'+str(z) for z in range(1,6)]
            tls = self.df[tl_cols]
            tls = _get_enumerated(tls)
            # Find the unique trajectories.
            cl_keys, cl_nums = np.unique(tls, axis=0, return_counts=True)
            # Save the results.
            self.cl_dfs = [] # List of dataframes of subjects with the same disease trajectory.
            for i in range(len(cl_nums)):
                curr_key = cl_keys[i, :]
                self.df.loc[(tls == curr_key).all(axis=1), 'Cluster'] = i
                self.cl_dfs.append(self.df.loc[(tls == curr_key).all(axis=1)])
            self.df_cols = self.df_cols.append(pd.Series({'Name':'Cluster', 'Mod':'Cluster'}), ignore_index=True)

        def write():
            """
            Saves the updated data of selected participants as df_sds.csv at directory_to_save.
            Save the updated data dictionary as df_cols_sds.csv.
            """
            self.df.to_csv(self.directory_to_save+'df_prepared.csv', index=False)
            self.df_cols.to_csv(self.directory_to_save+'df_cols_prepared.csv', index=False)

        def create_trainvaltest_splits():
            """
            Split data into train/val/test splits and save the indices.
            """   
            def _write(data, filename):
                """
                Helper function to write the indices.
                """
                with open(filename, 'wb') as output_file:
                    pickle.dump(data, output_file)
            # Create a folder called splits at directory_to_save.
            try:
                os.makedirs(self.directory_to_save+'splits/')
            except OSError:
                pass     
            # Main operation.
            for rt in range(200):
                # For each train/test split, we have 5 train/val splits for cross validation. We hold them in the lists below.
                indice_trains = [[] for z in range(5)]
                indice_vals = [[] for z in range(5)]
                indice_test = [[] for z in range(5)]
                # Split each cluster into train/val/test sets.
                for cl_df in self.cl_dfs: 
                    N = len(cl_df)
                    idxs = np.array(cl_df.index, dtype=np.int32)
                    np.random.shuffle(idxs)
                    idxs_train_val, idxs_test = idxs[:int(N*0.8)], idxs[int(N*0.8):] # 80/20 train/test split.
                    indice_test.append(copy.deepcopy(idxs_test))
                    for rv in range(5):
                        N = len(idxs_train_val)
                        np.random.shuffle(idxs_train_val)
                        idxs_train, idxs_val = idxs_train_val[:int(N*0.875)], idxs_train_val[int(N*0.875):] # 70/10 train/val split for cross validation.
                        indice_trains[rv].append(copy.deepcopy(idxs_train))
                        indice_vals[rv].append(copy.deepcopy(idxs_val))
                # Concatenate the indices from each cluster.
                indice_test = np.concatenate(indice_test)
                for rv in range(5):
                    indice_trains[rv] = np.concatenate(indice_trains[rv])
                    indice_vals[rv] = np.concatenate(indice_vals[rv])
                # Save the indices.
                for rv in range(5):
                    _write([indice_trains[rv], indice_vals[rv], indice_test], self.directory_to_save+'splits/tvt_'+str(rt)+'_'+str(rv)+'.txt')
            
        # Apply the methods.
        read()
        add_Cluster_column()
        write()
        create_trainvaltest_splits()
