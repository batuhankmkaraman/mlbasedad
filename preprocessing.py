import argparse

from preprocessing_steps.step1 import Step1
from preprocessing_steps.step2 import Step2
from preprocessing_steps.step3 import Step3
from preprocessing_steps.step4 import Step4
from preprocessing_steps.step5 import Step5

class Parser(argparse.ArgumentParser):
    """
    Custom argument parser for I/O parameters.

    Attributes:
        path_to_read (str): The directory that contains both ADNIMERGE_DICT.csv and ADNIMERGE.csv.
        r_test (float): Desired test set ratio (between 0 and 1).
        r_val (float): Desired validation set ratio (between 0 and 1).
        N_RT (int): Number of train/test splits to be generated.
        N_RV (int): Number of train/val splits to be generated for each train/test split.
        seed (int): Seed for reproducibility.
    """
    def __init__(self):
        super(Parser, self).__init__(description='Custom argument parser for I/O parameters.')
        self.add_argument('--path_to_read', default='./',
                type=str, help='The directory that contains both ADNIMERGE_DICT.csv and ADNIMERGE.csv.')
        self.add_argument('--r_test', default=0.2,
                type=float, help='Desired test set ratio (between 0 and 1)')
        self.add_argument('--r_val', default=0.2,
                type=float, help='Desired validation set ratio (between 0 and 1)')
        self.add_argument('--N_RT', default=200,
                type=int, help='Number of train/test splits to be generated.')
        self.add_argument('--N_RV', default=5,
                type=int, help='Number of train/val splits to be generated for each train/test split.')
        self.add_argument('--seed', default=1337,
                type=int, help='Seed for reproducibility.')
        
    def parse(self):
        """
        Parses the arguments and returns an object containing the input parameters.
        """
        args = self.parse_args()
        return args

class Preprocessing:
    """
    Object that applies all the preprocessing steps.
    """  
    def __init__(self, args):
        
        self.args = args
        
        self.step1 = Step1()
        self.step2 = Step2()
        self.step3 = Step3()
        self.step4 = Step4()
        self.step5 = Step5()
        
    def apply(self):
        """
        Apply every preprocessing step.

        Args:
            path_to_read (str): The directory that contains both ADNIMERGE_DICT.csv and ADNIMERGE.csv.
        """
        print('Starting preprocessing...')
        self.step1.apply_all(self.args)
        self.step2.apply_all(self.args)
        self.step3.apply_all(self.args)
        self.step4.apply_all(self.args)
        self.step5.apply_all(self.args)
        print('Preprocessing is complete.')
        
def main():
    """
    Main function that parses the arguments and applies all preprocessing steps.
    """
    args = Parser().parse()
    aq_allsteps = Preprocessing(args)
    aq_allsteps.apply()
    
if __name__ == '__main__':
    main()
