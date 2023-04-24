import os 

class DirectoryNavigator:
    """
    A utility class to manage directories for data, results, and tensorboard.

    Args:
        root (str): The root directory path.
        model_name (str): The name of the model.
        is_multi (bool): Indicates whether the model is multi-year.
        year (int): The year of the model.

    Attributes:
        root (str): The root directory path.
        exp_name (str): The name of the experiment.
        model_name (str): The name of the model.
    """

    def __init__(self, root, model_name, is_multi, year):
        """
        Constructor for the DirectoryNavigator class.

        Args:
            root (str): The root directory path.
            model_name (str): The name of the model.
            is_multi (bool): Indicates whether the model is multi-year.
            year (int): The year of the model.
        """
        self.root = root
        self.model_name = model_name
        
        # Set the experiment name based on whether the model is multi-year.
        if is_multi:
            self.exp_name = model_name
        else:
            self.exp_name = model_name + '_' + str(year)

    def get_data_dir(self, i_rt, i_rv=0):
        """
        Get the directory path for the data.

        Args:
            i_rt (int): Current train/test split index.
            i_rv (int, optional): Current train/val split index.

        Returns:
            str: The directory path for the data.
        """
        return self.root + 'rt' + str(i_rt) + '/' + 'rv' + str(i_rv) + '/'

    def get_training_results_dir(self, i_rt, i_rv, i_ri):
        """
        Get the directory path to write the training results.

        Args:
            i_rt (int): Current train/test split index.
            i_rv (int): Current train/val split index.
            i_ri (int): Current random initialization index.

        Returns:
            str: The directory path for the training results.
        """
        # Directory path for the training results is obtained by appending the result directory path to the data directory path.
        return self.get_data_dir(i_rt, i_rv) + 'ri' + str(i_ri) + '/' + self.exp_name + '/'

    def create_training_results_dir(self, i_rt, i_rv, i_ri):
        """
        Create a directory for the training results.

        Args:
            i_rt (int): Current train/test split index.
            i_rv (int): Current train/val split index.
            i_ri (int): Current random initialization index.
        """
        # Create the result directory if it does not exist.
        results_dir = self.get_training_results_dir(i_rt, i_rv, i_ri)
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)

    def get_tb_dir(self, i_rt, i_rv, i_ri):
        """
        Get the directory path for the tensorboard.

        Args:
            i_rt (int): Current train/test split index.
            i_rv (int): Current train/val split index.
            i_ri (int): Current random initialization index.

        Returns:
            str: The directory path for the tensorboard.
        """
        # Directory path for the tensorboard is obtained by appending the tensorboard directory path to the data directory path.
        return self.get_data_dir(i_rt, i_rv) + 'ri' + str(i_ri) + '/' + 'tensorboard/' + self.exp_name + '/'
            
    def get_eval_rt_dir(self, i_rt):
        """
        Get the directory path to write the evaluation results of a single train/test split.

        Args:
            i_rt (int): Current train/test split index.
            i_rv (int, optional): Current train/val split index.

        Returns:
            str: The directory path for the data.

        """
        return self.root + 'rt' + str(i_rt) + '/' 


    def get_eval_x_dir(self, x, i_rt):
        """
        Get the directory path to write the result of x'th evaluation.

        Args:
            i_rt (int): Current train/test split index.
            i_rv (int, optional): Current train/val split index.

        Returns:
            str: The directory path for the data.
        """
        return self.get_eval_rt_dir(i_rt) + 'eval_' +str(x) + '/' + self.exp_name + '/'


    def create_eval_x_dir(self, x, i_rt):
        """
        Create a directory for x'th evaluation.

        Args:
            i_rt (int): Current train/test split index.

        Returns:
            str: The directory path for the created directory.
        """
        # Create the result directory if it does not exist.
        results_dir = self.get_eval_x_dir(x, i_rt)
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
        return results_dir


    def get_stats_x_dir(self, x):
        """
        Get the directory path to store results for statistical analysis.

        Args:
            x (int): Evaluation index.

        Returns:
            str: The directory path for the data.
        """
        return self.root + 'stats_' +str(x)+ '/'


    def create_stats_x_dir(self, x):
        """
        Create a directory to store results for statistical analysis.

        Args:
            x (int): The ID of the data.

        Returns:
            str: The directory path for the created directory.
        """
        # Create the result directory if it does not exist.
        results_dir = self.get_stats_x_dir(x)
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
        return results_dir
