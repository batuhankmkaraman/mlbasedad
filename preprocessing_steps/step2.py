import pandas as pd
import numpy as np
import os

class Step2:
    def __init__(self):
        """
        Object that reads the outputs of Step1, performs the first stage of the participant selection and saves the outputs.
        """  
        
    def read(self, path_to_read):
        """
        Read the outputs of Step1. The operations will be applied to self.df.
        
        Args:
            path_to_read (str): The directory containing the ADNIMERGE_DICT.csv and ADNIMERGE.csv files.  
        """
        path_to_read = path_to_read
        self.df_raw = pd.read_csv(path_to_read+'preprocessing_outputs/step1.csv', low_memory=False)
        self.df_dict_raw = pd.read_csv(path_to_read+'preprocessing_outputs/step1_DICT.csv', low_memory=False)
        self.df = self.df_raw.copy()
        
    def drop_no_dx_visits(self):
        """
        Drop visits with no diagnosis.
        """
        self.df = self.df.dropna(subset=['DX']).reset_index(drop=True)
        
    def drop_AD_baseline_subjects(self):
        """
        Drop subjects who have Alzheimer's at baseline.
        """
        def operation(df):
            rids_to_drop = []
            for rid in np.unique(df['RID']):
                df_rid = df.loc[df['RID']==rid]
                if df_rid.loc[df_rid['VISCODE']=='bl', 'DX'].values == 'Dementia': 
                    rids_to_drop.append(rid)
            df = df.loc[~df['RID'].isin(rids_to_drop)].reset_index(drop=True)
            return df
        self.df = operation(self.df)
        
    def drop_reverter_subjects(self):
        """
        We drop a subject from our study if the subject was diagnosed as CN after being diagnosed as MCI, and/or if the subject was diagnosed as CN or MCI after being diagnosed as AD.
        """
        def operation(df):
            rids_to_drop = []
            for rid in np.unique(df['RID']):
                df_rid = df.loc[df['RID']==rid].sort_values(by=['M']).reset_index(drop=True)
                flag = False
                for i in range(len(df_rid)-1):
                    dx_curr = df_rid.loc[i, 'DX']
                    if dx_curr == 'MCI':
                        if 'CN' in df_rid.loc[i+1:, 'DX'].values:
                            flag = True
                            break
                    elif dx_curr == 'Dementia':
                        if ('CN' in df_rid.loc[i+1:, 'DX'].values) or ('MCI' in df_rid.loc[i+1:, 'DX'].values):
                            flag = True
                            break
                if flag:
                    rids_to_drop.append(rid)
            df = df.loc[~df['RID'].isin(rids_to_drop)].reset_index(drop=True)
            return df
        self.df = operation(self.df)

    def write(self, path_to_write):
        """
        Write the processed data and its dictionary at path_to_write.
        
        Args:
            path_to_write: Path to the output folder.
        """
        path_to_write += 'preprocessing_outputs/'
        if not os.path.exists(path_to_write):
            os.makedirs(path_to_write)
        self.df.to_csv(path_to_write+'step2.csv', index=False)
        self.df_dict_raw.to_csv(path_to_write+'step2_DICT.csv', index=False)
        
    def apply_all(self, args):
        """
        Apply the methods. 
        """
        self.read(args.path_to_read)
        self.drop_no_dx_visits()
        self.drop_AD_baseline_subjects()
        self.drop_reverter_subjects()
        self.write(args.path_to_read)
        print('Step 2 is complete.')
   