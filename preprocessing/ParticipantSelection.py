import pandas as pd
import numpy as np
import copy

class ParticipantSelection:
    def __init__(self, directory_to_read, directory_to_save):
        """
        Reads prepared data and its dictionary, performs participant selection, gets disease progression trajectories of subjects, updates the data dictionary, and saves the outputs.
        Args:
            directory_to_read : Directory of prepared data.
            directory_to_save : Directory to save the data of selected participants and updated data dictionary.
        Returns:
            Saves the data of selected participants as df_ps.csv.
            Saves the updated data dictionary as df_cols_ps.csv.
        """
        self.directory_to_read = directory_to_read
        self.directory_to_save = directory_to_save

        def read():
            """
            Read the prepared data.
            """
            self.df = pd.read_csv(self.directory_to_read+'df_prepared.csv', low_memory=False)
            self.df_cols = pd.read_csv(self.directory_to_read+'df_cols_prepared.csv', low_memory=False)

        def drop_no_dx():
            """
            Drop visits with no diagnosis.
            """
            self.df = self.df.dropna(subset=['DX'])

        def get_baseline_df():
            """
            Get baseline visits of CN baseline and MCI baseline subjects.
            """
            self.df_bl = copy.deepcopy(self.df)
            self.df_bl = self.df_bl.loc[self.df_bl['VISCODE']=='bl']
            self.df_bl = self.df_bl.loc[self.df_bl['DX'].isin(['CN', 'MCI'])]
            self.df_bl.reset_index(drop=True, inplace=True)

        def remove_reverter_subjects():
            """
            We remove a subject from our study if the subject was diagnosed as CN after being diagnosed as MCI, and/or if the subject was diagnosed as CN or MCI after being diagnosed as Dementia.
            """
            clean = []
            for rid_i in range(len(self.df_bl)):
                rid = self.df_bl.loc[rid_i, 'RID']
                df_rid = self.df.loc[self.df['RID']==rid].reset_index(drop=True)
                df_traj = list(df_rid['DX'].values) # traj: disease trajectory.
                flag = True
                for i in range(0, len(df_traj)-1):
                    dx_curr = df_traj[i]
                    if dx_curr == 'MCI':
                        if 'CN' in df_traj[i+1:]:
                            flag = False
                            break
                    elif dx_curr == 'Dementia':
                        if 'CN' in df_traj[i+1:] or 'MCI' in df_traj[i+1:]:
                            flag = False
                            break
                if flag == True:
                    clean.append(self.df_bl.loc[rid_i])
            clean = pd.DataFrame(clean).reset_index(drop=True)
            self.df_bl = clean
             
        def get_df_trajs():
            """
            Get the disease trajectory dataframe for the five-year follow-up horizon.
            isimpStab and isimpConv are initialized to indicate whether the diagnosis comes from a imputation operation. The actual imputation operation is performed at a later stage with a different function.
            Although we have implemeted an imputation method for stable subjects, we do not use it in our paper.
            isimpConv is used to indicate the diagnoses that we impute in the way we describe in the manuscript.
            """
            def _find_visit(df_rid, year):
                month = year*12
                df_month = copy.deepcopy(df_rid)
                df_month = df_month.loc[df_month['VISCODE']=='m'+str(month)]
                if len(df_month) !=0 :
                    df_month.reset_index(drop=True, inplace=True)
                    visit_month, visit_dx = df_month.loc[(df_month['Month_bl']-month).abs().argsort()[0], ['Month_bl', 'DX']]
                else: # Subjects may have converted in mid-year visits, i.e., visits at 6th, 18th... month.
                    df_month_only = df_rid.loc[1:, 'VISCODE'].astype(str).str[1:].astype(float)
                    df_month = df_rid.loc[1:].loc[df_month_only<month, ['Month_bl', 'DX']]
                    df_month = df_month.loc[1:].loc[month-12<df_month_only, ['Month_bl', 'DX']]
                    if df_rid.loc[0, 'DX'] == 'CN' and 'MCI' in df_month['DX'].values:
                        df_month = df_month.loc[df_month['DX']=='MCI']
                        visit_month, visit_dx = df_month.iloc[-1]
                        visit_month = year*12
                    elif df_rid.loc[0, 'DX'] == 'CN' and 'Dementia' in df_month['DX'].values:
                        df_month = df_month.loc[df_month['DX']=='Dementia']
                        visit_month, visit_dx = df_month.iloc[-1]
                        visit_month = year*12
                    elif df_rid.loc[0, 'DX'] == 'MCI' and 'Dementia' in df_month['DX'].values:
                        df_month = df_month.loc[df_month['DX']=='Dementia']
                        visit_month, visit_dx = df_month.iloc[-1]
                        visit_month = year*12
                    else:
                        visit_month, visit_dx = np.nan, np.nan
                return visit_month, visit_dx
            df_trajs = pd.DataFrame()
            isimpStab = pd.DataFrame(columns=['isimpStab_'+str(year) for year in range(1,6)])
            isimpConv = pd.DataFrame(columns=['isimpConv_'+str(year) for year in range(1,6)])
            for i in range(len(self.df_bl)):
                rid = self.df_bl.loc[i, 'RID']
                df_trajs.loc[i, 'RID'] = rid
                df_rid = self.df.loc[self.df['RID']==rid].reset_index(drop=True)
                df_trajs.loc[i, 'M_0'] = 0
                df_trajs.loc[i, 'DX_0'] = df_rid.loc[0, 'DX']
                for year in range(1, 6):
                    out = _find_visit(df_rid, year)
                    df_trajs.loc[i, 'M_'+str(year)] = out[0]
                    df_trajs.loc[i, 'DX_'+str(year)] = out[1]
                    isimpStab.loc[i, 'isimpStab_'+str(year)] = 0
                    isimpConv.loc[i, 'isimpConv_'+str(year)] = 0
            self.df_trajs = pd.concat([df_trajs, isimpStab, isimpConv], axis=1)
        
        def impute_stable_trajs_with_visits_after_year_5():
            """"
            Impute stable subjects' DX_5 column if they have a follow-up visit (in which they do not progress to the next disease stage) after year 5.
            """
            for dx_0 in ['CN', 'MCI']:
                for i in range(len(self.df_trajs)):
                    if self.df_trajs.loc[i, 'DX_0'] == dx_0:
                        rid = self.df_bl.loc[i, 'RID']
                        df_rid = self.df.loc[self.df['RID']==rid].reset_index(drop=True)
                        df_traj = df_rid[['VISCODE', 'DX']]
                        df_traj_future = []
                        flag = False
                        for j in range(1, len(df_traj)):
                            if int(df_traj.loc[j, 'VISCODE'][1:]) > 60:
                                flag = True
                            if flag:
                                df_traj_future.append(df_traj.loc[j])
                        if len(df_traj_future) != 0:
                            df_traj = pd.DataFrame(df_traj_future).reset_index(drop=True)
                            if dx_0 in list(df_traj['DX'].values) and pd.isna(self.df_trajs.loc[i, 'DX_5']):
                                self.df_trajs.loc[i, 'M_5'] = 60
                                self.df_trajs.loc[i, 'DX_5'] = dx_0
                                self.df_trajs.loc[i, 'isimpStab_5'] = 1
        
        def remove_no_follow_ups():
            """
            Remove subjects without any follow-up.
            """
            clean = []
            months = ['M_'+str(z) for z in range(1,6)]
            for i in range(len(self.df_trajs)):
                if pd.isna(self.df_trajs.loc[i, months]).all():
                    continue
                else:
                    clean.append(self.df_trajs.loc[i])
            clean = pd.DataFrame(clean).reset_index(drop=True)
            self.df_trajs = clean
        
        def remove_cn_to_dementias():
            """
            Remove subjects who converted from CN to Dementia in five years, since they are few in number.
            """
            clean = []
            dx_cols = ['DX_'+str(z) for z in range(6)]
            for i in range(len(self.df_trajs)):
                if 'CN' in list(self.df_trajs.loc[i, dx_cols].values) and 'Dementia' in list(self.df_trajs.loc[i, dx_cols].values):
                    continue
                else:
                    clean.append(self.df_trajs.loc[i])
            clean = pd.DataFrame(clean).reset_index(drop=True)
            self.df_trajs = clean

        def impute_stable_trajs():
            """
            Imputes a missing diagnosis if the subject is CN/MCI at baseline and is diagnosed as CN/MCI in a later follow-up visit.
            """
            for dx_bl in ['CN', 'MCI']:
                for i in range(len(self.self.df_trajs)):
                    if self.df_trajs.loc[i, 'dx_bl'] == dx_bl:
                            for year in range(4, 0, -1):
                                dx_future = self.df_trajs.loc[i, 'DX_'+str(year+1)]
                                if pd.isna(dx_future) == False:
                                    if dx_future == dx_bl and pd.isna(self.df_trajs.loc[i, 'DX_'+str(year)]):
                                        self.df_trajs.loc[i, 'M_'+str(year)] = year * 12
                                        self.df_trajs.loc[i, 'DX_'+str(year)] = dx_bl
                                        self.df_trajs.loc[i, 'isimpStab_'+str(year)] = 1
        
        def impute_converter_trajs():
            """
            Imputes a missing diagnosis if the subject is CN/MCI at baseline and is diagnosed as MCI/Dementia in an earlier follow-up visit.
            """
            for dx_bl, dx_conv in [['CN', 'MCI'], ['MCI', 'Dementia']]:
                for i in range(len(self.df_trajs)):
                    if self.df_trajs.loc[i, 'DX_0'] == dx_bl:
                            for year in range(2, 6):
                                dx_prev = self.df_trajs.loc[i, 'DX_'+str(year-1)]
                                if pd.isna(dx_prev) == False:
                                    if dx_prev == dx_conv and pd.isna(self.df_trajs.loc[i, 'DX_'+str(year)]):
                                        self.df_trajs.loc[i, 'M_'+str(year)] = year * 12
                                        self.df_trajs.loc[i, 'DX_'+str(year)] = dx_conv
                                        self.df_trajs.loc[i, 'isimpConv_'+str(year)] = 1

        def get_df_ps():
            """
            At this stage, participant selection and label imputation is complete. We create a new df, self.df_ps, for the selected participants.
            """
            self.df_bl = self.df_bl.loc[self.df_bl['RID'].isin(self.df_trajs['RID'])].reset_index(drop=True)
            trajs_cols = np.setdiff1d(list(self.df_trajs.columns), ['RID'])
            self.df_ps = pd.concat([self.df_bl, self.df_trajs[trajs_cols]], axis=1)
            
        def add_trajectory_labels():
            """
            Add the TL (trajectory Label) column for each follow-up year. As described in the manuscript, these trajectory labels will be used for 1)determining sample points' weights and 2)the stratified splitting of data into train, validation, and test sets.
            """
            s = [['CN', 'CN'],
                    ['CN', 'MCI'],
                    ['MCI', 'MCI'],
                    ['MCI', 'Dementia']]
            for i in range(len(self.df_ps)):
                for year in range(1, 6):
                    pair = list(self.df_ps.loc[i, ['DX_0', 'DX_'+str(year)]])
                    if pd.isna(pair).any() == False:
                        self.df_ps.loc[i, 'TL_'+str(year)] = s.index(pair)
                    else:
                        self.df_ps.loc[i, 'TL_'+str(year)] = pd.NA

        def get_updated_df_cols():
            """
            Get the new data dictionary self.df_cols_ps
            """
            cols_month = ['M_'+str(z) for z in range(6)]
            df_month = pd.DataFrame({'Name':cols_month, 'Mod':'Month'})
            cols_fdx = ['DX_'+str(z) for z in range(6)]
            df_fdx = pd.DataFrame({'Name':cols_fdx, 'Mod':'FDx'})
            cols_isimpStab = ['isimpStab_'+str(z) for z in range(1, 6)]
            df_isimpStab = pd.DataFrame({'Name':cols_isimpStab, 'Mod':'isimpStab'})
            cols_isimpConv = ['isimpConv_'+str(z) for z in range(1, 6)]
            df_isimpConv = pd.DataFrame({'Name':cols_isimpConv, 'Mod':'isimpConv'})
            cols_TrajLabel = ['TL_'+str(z) for z in range(1, 6)]
            df_TrajLabel = pd.DataFrame({'Name':cols_TrajLabel, 'Mod':'TrajLabel'})
            self.df_cols_ps = pd.concat([self.df_cols, df_month, df_fdx, df_isimpStab, df_isimpConv, df_TrajLabel])

        def write():
            """
            Save the selected participants' data as df_ps.csv at directory_to_save.
            Save the dictionary for columns of selected participants' data as df_cols_ps.csv at directory_to_save.
            """
            self.df_ps.to_csv(self.directory_to_save+'df_ps.csv', index=False)
            self.df_cols_ps.to_csv(self.directory_to_save+'df_cols_ps.csv', index=False)

        # Apply the methods.
        read()
        drop_no_dx()
        get_baseline_df()
        remove_reverter_subjects()
        get_df_trajs()
        # impute_stable_trajs_with_visits_after_year_5()
        remove_no_follow_ups()
        remove_cn_to_dementias()
        # impute_stable_trajs()
        impute_converter_trajs()
        get_df_ps()
        add_trajectory_labels()
        get_updated_df_cols()
        write()
