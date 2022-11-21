import pandas as pd
import random
import numpy as np
import pickle
import copy
import os

class Analyze:
    def __init__(self, directory_to_read):
        """
        Reads selected participants' data and its dictionary, and analyzes the data.
        Args:
            directory_to_read : Directory of selected participants' data.
        Returns:
            Prints summary statistics of the participants at baseline (similar to Table 1 in the manuscript).
            Prints the number of available subjects in each diagnostic group for annual follow-up visits (similar to Table 2 in the manuscript).
            Prints the degree of missingness (%) (similar to Table 3 in the manuscript).
        """
        self.directory_to_read = directory_to_read
    
    def read(self):
        """
        Read selected participants' data.
        """
        self.df = pd.read_csv(self.directory_to_read, low_memory=False)
        self.df_cols = pd.read_csv(self.directory_to_read, low_memory=False)

    def print_table_1(self):
        """
        Print summary statistics of the participants at baseline (similar to Table 1 in the manuscript).
        """
        print('Table 1:')
        for feat_name in ['PTGENDER', 'AGE', 'PTEDUCAT', 'APOE4', 'CDRSB', 'MMSE']:
            table_row = []
            for dx_bl in ['CN', 'MCI']:
                df_bl = self.df.loc[self.df['DX']==dx_bl]
                if feat_name in ['PTGENDER', 'APOE4']:
                    sub_df = df_bl.loc[:, feat_name]
                    keys, vals = np.unique(sub_df, return_counts=True)
                    table_row.append(str(vals))
                else:
                    sub_df = df_bl.loc[:, feat_name]
                    mean_value = str(np.round(sub_df.mean(), 2))
                    std_value = str(np.round(sub_df.std(), 2))
                    table_row.append(mean_value+' \pm '+std_value)
            if feat_name in ['PTGENDER', 'APOE4']:
                print(feat_name+' '+str(keys)+': '+str(table_row))
            else:
                print(feat_name+': '+str(table_row))

    def print_table_2(self):
        """
        Prints the number of available subjects in each diagnostic group for annual follow-up visits (similar to Table 2 in the manuscript).
        """
        print('Table 2:')
        for year in range(1, 6):
            table_row = []
            for dx_bl in ['CN', 'MCI']:
                tbl = []
                df_bl = self.df.loc[self.df['DX']==dx_bl]
                # Declare the label imputation choice, if any.
                # df_bl = self.df_bl.loc[self.df_bl['isimpStab_'+str(year)]==0]
                # df_bl = self.df_bl.loc[self.df_bl['isimpConv_'+str(year)]==0]
                for dx in ['CN', 'MCI', 'Dementia']:
                    tbl.append(np.sum(df_bl['DX_'+str(year)]==dx))
                table_row.append(tbl)
            print(table_row)

    def print_table_3(self):
        """
        Prints the degree of missingness (%) (similar to Table 3 in the manuscript).
        """
        print('Table 3:')
        for col_name, data_id in [['Mod', 'Cli'], ['Mod', 'Cog'], ['Mod', 'Csf'], ['Mod', 'Mri'], ['Name', 'FDG'], ['Name', 'AV45'], ['Name', 'PIB']]:
            table_row = []
            for dx_bl in ['CN', 'MCI']:
                df_bl = self.df.loc[self.df['DX']==dx_bl]
                sub_df = df_bl.loc[:, self.df_cols.loc[self.df_cols[col_name] == data_id, 'Name']]
                table_row.append(np.round(np.sum(np.sum(pd.isna(sub_df)))/len(sub_df)/len(sub_df.columns)*100, 2))
            print(data_id+': '+str(table_row))

    # Apply the methods.
    read()
    print_table_1()
    print_table_2()
    print_table_3()
