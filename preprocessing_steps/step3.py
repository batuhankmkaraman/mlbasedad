import pandas as pd
import numpy as np
import os

class Step3:
    def __init__(self):
        """
        Initializes an object that reads the outputs of Step2, performs the second stage of participant selection and label imputation, updates the data dictionary, and saves the outputs.
        """  
        
    def read(self, path_to_read):
        """
        Reads the outputs of Step2 and stores the data in the class object for further operations.
        
        Args:
            path_to_read (str): The directory containing the ADNIMERGE_DICT.csv and ADNIMERGE.csv files.  
        """
        path_to_read = path_to_read
        self.df_raw = pd.read_csv(path_to_read+'preprocessing_outputs/step2.csv', low_memory=False)
        self.df_dict_raw = pd.read_csv(path_to_read+'preprocessing_outputs/step2_DICT.csv', low_memory=False)
        self.df = self.df_raw.copy()
        self.df_dict = self.df_dict_raw.copy()
        
    def extract_longi_labels_and_months(self):
        """
        Extract the longitudinal labels of each subject. Column names are visit months and each row is a subject.
        """
        def operation(df):
            
            longi_labels = pd.DataFrame()
            longi_months = pd.DataFrame()
            for i, rid in enumerate(np.unique(df['RID'])):
                df_rid = df.loc[df['RID']==rid].sort_values(by=['M']).reset_index(drop=True)
                for _, visit in df_rid.iterrows():
                    longi_labels.loc[i, int(visit['M'])] = visit['DX']
                    longi_months.loc[i, int(visit['M'])] = visit['Month_bl']
                    
            longi_labels = longi_labels[list(np.sort(list(longi_labels.columns)))] # Sort the follow-up timepoints.
            longi_labels = longi_labels.rename(columns=dict(zip(longi_labels.columns, ['m'+str(z) for z in longi_labels.columns])))
            longi_labels.insert(0, 'RID', np.unique(df['RID']))
            
            longi_months = longi_months[list(np.sort(list(longi_months.columns)))] # Sort the follow-up timepoints.
            longi_months = longi_months.rename(columns=dict(zip(longi_months.columns, ['m'+str(z) for z in longi_months.columns])))
            longi_months.insert(0, 'RID', np.unique(df['RID']))
            
            return longi_labels, longi_months
        self.longi_labels, self.longi_months = operation(self.df)
        
        
    def perform_ad_label_imputation(self):
        """
        If a subject is AD at a timepoint, then they are AD for all future timepoints.
        """
        def operation(longi_labels, longi_months):
            # Label imputation.
            for i in range(len(longi_labels)):
                if longi_labels.loc[i].isin(['Dementia']).any():
                    first_ad_index = list(longi_labels.loc[i, :] == 'Dementia').index(True)
                    longi_labels.iloc[i, first_ad_index:] = 'Dementia'
                    
            # Followup month imputation.        
            for col in longi_labels.columns[1:]:
                # create a boolean mask to identify missing values in longi_months but not in longi_labels
                missing_mask = longi_months[col].isnull() & longi_labels[col].notnull()
                # fill the missing values with the column name
                longi_months.loc[missing_mask, col] = float(col[1:])

            return longi_labels, longi_months
        self.longi_labels, self.longi_months = operation(self.longi_labels, self.longi_months)
        
        
    def perform_mci_label_imputation(self):
        """
        If a CN baseline subject is MCI at a timepoint, then they are MCI for all future timepoints until they convert to Dementia.
        """
        def operation(longi_labels, longi_months):
            # Label imputation.
            for i in range(len(longi_labels)):
                if longi_labels.loc[i, 'm0'] == 'CN':
                    if longi_labels.loc[i].isin(['MCI']).any():
                        first_mci_index = list(longi_labels.loc[i, :] == 'MCI').index(True)
                        if longi_labels.loc[i].isin(['Dementia']).any():
                            first_ad_index = list(longi_labels.loc[i, :] == 'Dementia').index(True)
                        else:
                            first_ad_index = len(longi_labels.loc[i, :])
                        longi_labels.iloc[i, first_mci_index:first_ad_index] = 'MCI'
                        
            # Followup month imputation.        
            for col in longi_labels.columns[1:]:
                # create a boolean mask to identify missing values in longi_months but not in longi_labels
                missing_mask = longi_months[col].isnull() & longi_labels[col].notnull()
                # fill the missing values with the column name
                longi_months.loc[missing_mask, col] = float(col[1:])

            return longi_labels, longi_months
        self.longi_labels, self.longi_months = operation(self.longi_labels, self.longi_months)
        
                        
    def extract_annual_labels_and_months_up_to_year_n(self, n=5):
        """
        We only use annual labels for training/testing.
        
        Args:
            n: Time horizon (inclusive, in years)
        """
        self.longi_labels = self.longi_labels[['RID'] + ['m'+str(year*12) for year in range(n+1)]]
        self.longi_months = self.longi_months[['RID'] + ['m'+str(year*12) for year in range(n+1)]]
        for year in range(n+1):
            self.longi_labels = self.longi_labels.rename({'m'+str(year*12):'FDX_'+str(year)}, axis=1)
            self.longi_months = self.longi_months.rename({'m'+str(year*12):'FMonth_'+str(year)}, axis=1)
        
    
    def drop_no_followup_subjects(self):
        """
        Drop subjects without any follow-up visit.
        """
        def operation(longi_labels, longi_months):
            rids_to_drop = []
            for i in range(len(longi_labels)):
                if not (longi_labels.iloc[i, 2:]==longi_labels.iloc[i, 2:]).any():
                    rids_to_drop.append(longi_labels.loc[i, 'RID'])
            df_labels = longi_labels.loc[~longi_labels['RID'].isin(rids_to_drop)].reset_index(drop=True)
            df_months = longi_months.loc[~longi_months['RID'].isin(rids_to_drop)].reset_index(drop=True)
            return df_labels, df_months
        self.longi_labels, self.longi_months = operation(self.longi_labels, self.longi_months)
        
        
    def drop_CN_to_AD_s(self):
        """
        Drop subjects who converted from CN to Dementia in the first five years, since they are few in number.
        """
        def operation(longi_labels, longi_months):
            rids_to_drop = []
            for i in range(len(longi_labels)):
                if ('CN' in longi_labels.loc[i].values) and ('Dementia' in longi_labels.loc[i].values):
                    rids_to_drop.append(longi_labels.loc[i, 'RID'])
            df_labels = longi_labels.loc[~longi_labels['RID'].isin(rids_to_drop)].reset_index(drop=True)
            df_months = longi_months.loc[~longi_labels['RID'].isin(rids_to_drop)].reset_index(drop=True)
            return df_labels, df_months
        self.longi_labels, self.longi_months = operation(self.longi_labels, self.longi_months) 
        
        
    def add_trajectory_labels(self):
        """
        These labels will be used for creating stratified train/val/test splits. We need stratification due to the heavily unbalanced nature of the problem.
        """
        def operation(longi_labelss):
            longi_labels = longi_labelss.copy()
            longi_labels = longi_labels.fillna('Unknown')
            longi_labels = longi_labels.replace({
                                            'CN':0,
                                            'MCI':1,
                                            'Dementia':2,
                                            'Unknown':-4
                                            })
            keys, indices= np.unique(longi_labels.iloc[:, 1:], axis=0, return_inverse=True)
            longi_labelss['TRAJ_LABEL'] = indices
            return longi_labelss
        self.longi_labels = operation(self.longi_labels)
        
        
    def drop_subjects_with_very_rare_trajs(self):
        """
        We drop subjects with a TRAJ_LABEL does not occur at least 3 times in the dataset, since we would not be able to stratify such a TRAJ_LABEL to train/val/test splits.
        """
        def operation(longi_labels, longi_months):
            keys, counts = np.unique(longi_labels['TRAJ_LABEL'], return_counts=True)
            longi_labels = longi_labels.loc[longi_labels['TRAJ_LABEL'].isin(keys[np.where(counts>=3)[0]])].reset_index(drop=True)
            longi_months = longi_months.loc[longi_months['RID'].isin(longi_labels['RID'])].reset_index(drop=True)
            return longi_labels, longi_months
        self.longi_labels, self.longi_months = operation(self.longi_labels, self.longi_months)
        """
        Then, We re-map the TRAJ_LABEL values so that they range from 0 to K-1 where K is number of unique trajectories.
        """
        def operation(longi_labels):
            keys = np.unique(longi_labels['TRAJ_LABEL'])
            for i in range(len(longi_labels)):
                longi_labels.loc[i, 'TRAJ_LABEL'] = np.where(keys==longi_labels.loc[i, 'TRAJ_LABEL'])[0]
            return longi_labels
        self.longi_labels = operation(self.longi_labels)
        
        
    def create_the_acquired_df_and_its_dict(self):
        """
        We are done with participant selection. We can now create the finalized version of acquired dataset with features at the baseline and visit month/visit dx pairs of followup visits.
        """
        def operation(df, df_dict, longi_labels, longi_months):
            df = df.loc[df['RID'].isin(longi_labels['RID'])]
            df_bl = df.loc[df['VISCODE']=='bl'].reset_index(drop=True)
            
            longi_labels = longi_labels.sort_values('RID').reset_index(drop=True)
            df_bl = df_bl.sort_values('RID').reset_index(drop=True) 

            acquired_df = pd.concat([df_bl, longi_months.iloc[:, 1:], longi_labels.iloc[:, 1:]], axis=1)

            longi_months_dict = pd.DataFrame(columns=df_dict.columns)
            longi_labels_dict = pd.DataFrame(columns=df_dict.columns)
            for i, (col_month, col_label) in enumerate(zip(longi_months.columns[1:], longi_labels.columns[1:-1])):
                longi_months_dict.loc[i, 'FLDNAME'] = col_month
                longi_months_dict.loc[i, 'TEXT'] = 'Exact month of the follow-up visit of year '+col_month.split('_')[1]
                
                longi_labels_dict.loc[i, 'FLDNAME'] = col_label
                longi_labels_dict.loc[i, 'TEXT'] = 'Diagnosis of the follow-up visit of year '+col_label.split('_')[1]
                
            longi_months_dict[['Modality', 'NumCat', 'dtype']] = 'FollowupMonth', 'Num', 'float64'

            longi_labels_dict[['Modality', 'NumCat', 'dtype']] = 'FollowupDX', 'Cat', 'object'
            longi_labels_dict.loc[len(longi_labels_dict)] = ['TRAJ_LABEL', 'Other', 'Cat', 
                                                            'Disease projection trajectory label', pd.NA]
                
            acquired_df_dict = pd.concat([df_dict, longi_months_dict, longi_labels_dict]).reset_index(drop=True)
            return acquired_df, acquired_df_dict
        self.acquired_df, self.acquired_df_dict = operation(self.df, self.df_dict, self.longi_labels, self.longi_months)

    def write(self, path_to_write):
        """
        Write the acquired df and its dictionary at path_to_write.
        
        Args:
            path_to_write: Path to the output folder.
        """
        path_to_write += 'preprocessing_outputs/'
        if not os.path.exists(path_to_write):
            os.makedirs(path_to_write)
        self.acquired_df.to_csv(path_to_write+'step3.csv', index=False)
        self.acquired_df_dict.to_csv(path_to_write+'step3_DICT.csv', index=False)
        
    def apply_all(self, args):
        """
        Apply the methods. 
        """
        self.read(args.path_to_read)
        self.extract_longi_labels_and_months()
        self.perform_ad_label_imputation()
        self.perform_mci_label_imputation()
        self.extract_annual_labels_and_months_up_to_year_n(n=5)
        self.drop_no_followup_subjects()
        self.drop_CN_to_AD_s()
        self.add_trajectory_labels()
        self.drop_subjects_with_very_rare_trajs()
        self.create_the_acquired_df_and_its_dict()
        self.write(args.path_to_read)
        print('Step 3 is complete.')

   
