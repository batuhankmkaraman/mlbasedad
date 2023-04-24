import pandas as pd
import numpy as np
import os

class Step1:
    def __init__(self):
        """
        Object that reads ADNI data, cleans it up, organizes it for the following stage of data preprocessing, and writes the prepared data and its dictionary.
        """  
        
    def read(self, path_to_read):
        """
        Read the raw data and initialize self.df, which will be used for further processing.

        Args:
            path_to_read (str): The directory containing the ADNIMERGE_DICT.csv and ADNIMERGE.csv files.  
        """
        # set the path to the directory containing the data files
        path_to_read = path_to_read

        # read in the ADNIMERGE.csv file and store as self.df_raw
        self.df_raw = pd.read_csv(path_to_read+'ADNIMERGE.csv', low_memory=False)

        # read in the ADNIMERGE_DICT.csv file and store as self.df_dict_raw
        self.df_dict_raw = pd.read_csv(path_to_read+'ADNIMERGE_DICT.csv', low_memory=False)

        # make a copy of self.df_raw and store as self.df for further processing
        self.df = self.df_raw.copy()

    def create_df_dict(self):
        """
        Initialize the dictionary for the columns of interest.
        
        Returns:
            None
        
        Dictionary structure:
        - Modality: Modality of the column.
        - NumCat: Indicates whether the feature is numerical or categorical.
        - 'DNA': Does Not Apply.
        """
        # 0. id 
        df_id = pd.DataFrame({'FLDNAME':['RID'
                                         ], 
                                            'Modality':'ID', 'NumCat':'DNA'})
        # 1. other
        df_other = pd.DataFrame({'FLDNAME':['COLPROT', 
                                            'ORIGPROT', 
                                            'PTID', 
                                            'SITE', 
                                            'VISCODE', 
                                            'EXAMDATE', 
                                            'FLDSTRENG', 
                                            'FSVERSION', 
                                            'IMAGEUID', 
                                            'Years_bl',
                                            'Month_bl',
                                            'Month',
                                            'M',
                                            'update_stamp'
                                            ], 
                                                'Modality':'Other', 'NumCat':'DNA'})
        # 2. demog
        df_demog = pd.DataFrame({'FLDNAME':['AGE', 
                                            'PTGENDER', 
                                            'PTEDUCAT', 
                                            'PTETHCAT', 
                                            'PTRACCAT', 
                                            'PTMARRY'
                                            ], 
                                                'Modality':'Demographic', 'NumCat':'Cat'})
        df_demog.loc[0, 'NumCat'] = 'Num' 
        df_demog.loc[2, 'NumCat'] = 'Num' 
        # 3. gene
        df_gene = pd.DataFrame({'FLDNAME':['APOE4'
                                           ], 
                                                'Modality':'Genetic', 'NumCat':'Cat'})
        # 5. pet
        df_pet = pd.DataFrame({'FLDNAME':['FDG', 
                                            'PIB', 
                                            'AV45',
                                            ], 
                                                'Modality':'PET', 'NumCat':'Num'})
        # 4. csf
        df_csf = pd.DataFrame({'FLDNAME':['ABETA', 
                                            'TAU', 
                                            'PTAU'
                                            ], 
                                                'Modality':'CSF', 'NumCat':'Num'})
        # 7. cli
        df_cli = pd.DataFrame({'FLDNAME':['CDRSB', # Ordinal.
                                            'FAQ', 
                                            'EcogPtMem', # All Ecogs are ordinal.
                                            'EcogPtLang',
                                            'EcogPtVisspat',
                                            'EcogPtPlan',
                                            'EcogPtOrgan',
                                            'EcogPtDivatt',
                                            'EcogPtTotal',
                                            'EcogSPMem',
                                            'EcogSPLang',
                                            'EcogSPVisspat',
                                            'EcogSPPlan',
                                            'EcogSPOrgan',
                                            'EcogSPDivatt',
                                            'EcogSPTotal'
                                            ], 
                                                'Modality':'Clinical', 'NumCat':'Num'})
        # 8. cog
        df_cog = pd.DataFrame({'FLDNAME':['ADAS11', 
                                            'ADAS13',
                                            'ADASQ4',
                                            'MMSE', 
                                            'RAVLT_immediate', 
                                            'RAVLT_learning',
                                            'RAVLT_forgetting',
                                            'RAVLT_perc_forgetting',
                                            'LDELTOTAL',
                                            'DIGITSCOR',
                                            'TRABSCOR',
                                            'MOCA',
                                            'mPACCdigit',
                                            'mPACCtrailsB'
                                            ], 
                                                'Modality':'Cognitive', 'NumCat':'Num'})
        # 6. mri
        df_mri = pd.DataFrame({'FLDNAME':['Ventricles', 
                                            'Hippocampus', 
                                            'WholeBrain', 
                                            'Entorhinal', 
                                            'Fusiform', 
                                            'MidTemp', 
                                            'ICV'
                                            ], 
                                                'Modality':'MRI', 'NumCat':'Num'})
        # 9. dx
        df_dx = pd.DataFrame({'FLDNAME':['DX'
                                         ], 
                                            'Modality':'Diagnosis', 'NumCat':'Cat'})
        # Concatenate.
        df_dict = pd.concat([df_id, 
                                    df_other, 
                                    df_demog, 
                                    df_gene, 
                                    df_cli, 
                                    df_cog, 
                                    df_csf, 
                                    df_pet, 
                                    df_mri, 
                                    df_dx]).reset_index(drop=True)
        # Save the dict.
        self.df_dict = df_dict
        
    def merge_df_dict_with_raw_dict(self):
        """
        Updates the 'TEXT' column in the df_dict DataFrame by matching the 'FLDNAME' column in df_dict with the 
        'FLDNAME' column in df_dict_raw and copying over the corresponding 'TEXT' value. If a match is not found, 
        the 'TEXT' value is set to 'Empty in original dict'.
        """
        def operations(df_dict, df_dict_raw):
            for i in range(len(df_dict)):
                fldname = df_dict.loc[i, 'FLDNAME']
                if fldname in df_dict_raw['FLDNAME'].values:
                    text = df_dict_raw.loc[df_dict_raw['FLDNAME'] == fldname, 'TEXT'].values[0]
                    df_dict.loc[i, 'TEXT'] = text
                else:
                    df_dict.loc[i, 'TEXT'] = 'Empty in original dict'
            return df_dict
        self.df_dict = operations(self.df_dict, self.df_dict_raw)
        
    def get_columns_of_interest(self):
        """
        This function filters the input DataFrame to only keep the columns specified in the dictionary.
        """
        # Extract the fieldnames from the dictionary
        fieldnames = self.df_dict['FLDNAME']

        # Filter the input DataFrame to only keep the desired columns
        self.df = self.df[fieldnames]

    def fix_terminology_in_visitcodes(self):
        """
        Fix the VISCODE column so that the visit codes are either 'bl' or 'mX' where X is the follow-up month.
        """
        self.df = self.df.loc[self.df['VISCODE']!='m0'].reset_index(drop=True) # There is a single row which has 'm0' as the visitcode and it's an empty visit. So, we just drop it.
        self.df.replace({'y1':'m12'}, inplace=True) # It looks like this is fixed by ADNI but we keep it here for convenience.
            
    def fix_values_in_csf_columns(self):
        """
        CSF data is censored. When the measurement is too high or too low, the cell data contains "<" or ">". 
        We remove "<" and ">" signs and decrease/incease the numerical value by 1, respectively.
        """
        def operation(df):
            csf_cols = ['ABETA', 'TAU', 'PTAU']
            for col in csf_cols:
                # 
                mask = df[col].str.contains('<').fillna(False)
                df.loc[mask, col] = df.loc[mask, col].str.replace('<', '').astype(float) - 1.
                #
                mask = df[col].str.contains('>').fillna(False)
                df.loc[mask, col] = df.loc[mask, col].str.replace('>', '').astype(float) + 1.
            return df
        self.df = operation(self.df)
        
    def fix_dtype_in_csf_columns(self):
        """
        CSF columns come as string columns. We make them floats here.
        """
        def operation(df):
            for col in ['ABETA', 'TAU', 'PTAU']:
                df[col] = df[col].fillna(np.nan)
                df[col] = df[col].astype(float)
            return df
        self.df = operation(self.df)
        
    def fix_missingness_in_categorical_columns(self):
        """
        Replace 'Unknown' with None.
        """
        self.df.replace({'Unknown': None}, inplace=True)

    def add_dtype_to_df_dict(self):
        """
        Add python dtype of the columns to the data dictionary.
        """
        for col in self.df.columns:
            self.df_dict.loc[self.df_dict['FLDNAME']==col, 'dtype'] = str(self.df[col].dtype) 
        
    def write(self, path_to_write):
        """
        Write the processed data and its dictionary at path_to_write.
        
        Args:
            path_to_write: Path to the output folder.
        """
        path_to_write += 'preprocessing_outputs/'
        if not os.path.exists(path_to_write):
            os.makedirs(path_to_write)
        self.df.to_csv(path_to_write+'step1.csv', index=False)
        self.df_dict.to_csv(path_to_write+'step1_DICT.csv', index=False)
        
    def apply_all(self, args):
        """
        Apply the methods. 
        """
        self.read(args.path_to_read)
        self.create_df_dict()
        self.merge_df_dict_with_raw_dict()
        self.get_columns_of_interest()
        self.fix_terminology_in_visitcodes()
        self.fix_values_in_csf_columns()
        self.fix_dtype_in_csf_columns()
        self.fix_missingness_in_categorical_columns()
        self.add_dtype_to_df_dict()
        self.write(args.path_to_read)
        print('Step 1 is complete.')
   