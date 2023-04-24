import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve

def get_bacc(y_true, y_pred, weight):
    """
    Compute the balanced accuracy score.

    Parameters:
    -----------
    y_true : array-like of shape (n_samples,)
        True labels of samples.
    y_pred : array-like of shape (n_samples, n_classes)
        Predicted probabilities for each class.
    weight : array-like of shape (n_samples,)
        Sample weights.

    Returns:
    --------
    bacc : float
        Balanced accuracy score.
    """
    # Normalize the weight based on the number of samples
    weight *= len(y_true) / np.sum(weight)
    
    # Get a boolean array indicating whether the prediction is correct or not
    correct_preds = (y_true == np.argmax(y_pred, axis=1))
    
    # Calculate the weighted number of correct predictions for each sample
    weighted_correct_preds = weight * correct_preds
    
    # Compute the average of the weighted correct predictions over all samples
    bacc = np.mean(weighted_correct_preds)
    
    # Scale the balance accuracy value to percentage.
    return bacc * 100


def get_rocauc(y_true, y_pred, dx_bl):
    """
    Calculate one-vs-rest the area under the receiver operating characteristic curve (AUC-ROC).
    For CN baseline, the positive class is MCI.
    For MCI baseline, the positive class is Dementia.

    Args:
        y_true (array-like): True binary labels.
        y_pred (array-like): Predicted probabilities or confidence scores for the positive class.
        dx_bl (int): Baseline diagnosis code. 0 for CN-baseline and 1 for MCI-baseline.

    Returns:
        float: The ROC_AUC score.

    """
    
    if dx_bl == 0:
        y_pred = y_pred[:, 1]  # Get the predicted probabilities the positive class.
        y_true = (y_true == 1).astype(int) # Get the one-vs-rest labels for the postive class.
    elif dx_bl == 1:
        y_pred = y_pred[:, 2]   # Get the predicted probabilities the positive class.
        y_true = (y_true == 2).astype(int)  # Get the one-vs-rest labels for the postive class.
        
    # Calculate the ROC_AUC score.
    roc_auc = roc_auc_score(y_true, y_pred) 
    
    return roc_auc * 100


def get_performance_metrics(df):
    """
    Calculates balanced accuracy (BAcc) and area under the receiver operating characteristic curve (ROCAUC) 
    for each follow-up year and baseline diagnostic group (CN or MCI) as well as the average over all years 
    and diagnostic groups.
    
    Parameters:
    df (pandas.DataFrame): Dataframe containing the true follow-up diagnoses (FollowupDX), predicted probabilities 
                            (Pred), and sample weights (SampleWeight) for each subject.
    
    Returns:
    metrics (dict): Dictionary containing the calculated metrics.
    """
    
    # Calculate metrics.
    metrics = {}

    # Loop over the two baseline diagnostic groups (CN and MCI).
    for dx_bl, dx_bl_name in zip([0, 1], ['CN', 'MCI']):
        # Subset the dataframe to include only subjects with the current baseline diagnosis.
        df_bl = df.loc[df['BaselineDX']==dx_bl]
        
        # Initialize lists to store BAcc and ROCAUC for each follow-up year.
        baccs = []
        rocaucs = []
        
        # Loop over the unique follow-up years.
        for year in np.unique(df['FollowupYear']):
            # Subset the dataframe to include only subjects with the current baseline diagnosis and follow-up year.
            year_df = df_bl.loc[df_bl['FollowupYear']==year]
            
            # Extract the true follow-up diagnoses, predicted probabilities, and sample weights.
            y_true = year_df['FollowupDX'].values
            y_pred = np.vstack(year_df['Pred'].values)
            weight = year_df['SampleWeight'].values
            
            # Calculate the BAcc and ROCAUC for the current year and baseline diagnosis.
            bacc = get_bacc(y_true, y_pred, weight)
            rocauc = get_rocauc(y_true, y_pred, dx_bl)
            
            # Add the BAcc and ROCAUC to the metrics dictionary with appropriate keys.
            metrics['BAcc_'+dx_bl_name+'_'+str(int(year))] = bacc
            metrics['ROCAUC_'+dx_bl_name+'_'+str(int(year))] = rocauc
            
            # Append the BAcc and ROCAUC to the lists.
            baccs.append(bacc)
            rocaucs.append(rocauc)
    
        # Calculate the average BAcc and ROCAUC over all follow-up years for the current baseline diagnosis.
        metrics['Avg_BAcc_'+dx_bl_name] = np.mean(baccs)
        metrics['Avg_ROCAUC_'+dx_bl_name] = np.mean(rocaucs)
    
    # Calculate the average BAcc and ROCAUC over all follow-up years and both baseline diagnostic groups.
    metrics['Avg_BAcc'] = 0.5*(metrics['Avg_BAcc_CN']+metrics['Avg_BAcc_MCI'])
    metrics['Avg_ROCAUC'] = 0.5*(metrics['Avg_ROCAUC_CN']+metrics['Avg_ROCAUC_MCI'])
            
    return metrics


def get_roc_aucs(df):
    """
    Computes the area under the ROC curve (ROCAUC) for each follow-up year and baseline diagnosis (CN or MCI).

    Args:
        df (pandas.DataFrame): DataFrame containing the true follow-up diagnosis (FollowupDX), predicted probabilities (Pred),
            and sample weights (SampleWeight) for each subject at each follow-up year and baseline diagnosis.

    Returns:
        dict: A dictionary containing the ROCAUC values for each follow-up year and baseline diagnosis.
    """

    # Initialize metrics dictionary
    rocaucs = {}

    # Loop through each baseline diagnosis (CN or MCI)
    for dx_bl, dx_bl_name in zip([0, 1], ['CN', 'MCI']):
        # Filter dataframe by baseline diagnosis
        df_bl = df.loc[df['BaselineDX']==dx_bl]
        # Loop through each follow-up year
        for year in np.unique(df['FollowupYear']):
            # Filter dataframe by follow-up year
            year_df = df_bl.loc[df_bl['FollowupYear']==year]
            
            # Extract true follow-up diagnoses, predicted probabilities, and sample weights
            y_true = year_df['FollowupDX'].values
            y_pred = np.vstack(year_df['Pred'].values)
            
            # Compute ROCAUC
            rocauc = roc_auc_score(y_true, y_pred[:, dx_bl])
            
            # Add ROCAUC value to metrics dictionary
            rocaucs['ROCAUC_'+dx_bl_name+'_'+str(int(year))] = rocauc
            
    return rocaucs


def get_roc_curve(y_true, y_pred, dx_bl):
    """
    Computes the ROC curve for a given diagnosis group (CN or MCI) and a given follow-up year.

    Args:
        y_true (numpy.ndarray): True follow-up diagnosis for each subject.
        y_pred (numpy.ndarray): Predicted probability of each subject belonging to each follow-up diagnosis group.
        dx_bl (int): Baseline diagnosis group (0 for CN, 1 for MCI).

    Returns:
        numpy.ndarray: Interpolated true positive rate (TPR) for the ROC curve at each false positive rate (FPR) value.
    """
    
    if dx_bl == 0:
        # For CN baseline diagnosis, use the probability of MCI diagnosis.
        y_pred = y_pred[:, 0] 
        y_true = (y_true == 0).astype(int)
        y_pred = 1 - y_pred
        y_true = 1 - y_true
    if dx_bl == 1:
        # For MCI baseline diagnosis, use the probability of AD diagnosis.
        y_pred = y_pred[:, 2]
        y_true = (y_true == 2).astype(int)
        
    # Compute the FPR, TPR, and thresholds for the ROC curve.
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)  
    
    # Interpolate the TPR values for a fixed set of FPR values (0 to 1 in increments of 0.01).
    mean_fpr = np.linspace(0, 1, 100)     
    interp_tpr = np.interp(mean_fpr, fpr, tpr)
    
    # Set the first TPR value to 0 to ensure the curve starts at (0, 0).
    interp_tpr[0] = 0.0
    
    # Scale TPR values to percentages.
    return interp_tpr * 100


def get_roc_curves(df):
    """
    Computes the ROC curves for each follow-up year and baseline diagnosis (CN or MCI).

    Args:
        df (pandas.DataFrame): DataFrame containing the true follow-up diagnosis (FollowupDX), predicted probabilities (Pred),
            and sample weights (SampleWeight) for each subject at each follow-up year and baseline diagnosis.

    Returns:
        dict: A dictionary containing the interpolated true positive rates (TPR) for each FPR value for each ROC curve.
    """

    # Initialize dictionary to store ROC curves.
    roc_curves = {}

    # Loop through each baseline diagnosis (CN or MCI)
    for dx_bl, dx_bl_name in zip([0, 1], ['CN', 'MCI']):
        # Filter dataframe by baseline diagnosis
        df_bl = df.loc[df['BaselineDX']==dx_bl]
        # Loop through each follow-up year
        for year in np.unique(df['FollowupYear']):
            # Filter dataframe by follow-up year
            year_df = df_bl.loc[df_bl['FollowupYear']==year]
            
            # Extract true follow-up diagnoses and predicted probabilities
            y_true = year_df['FollowupDX'].values
            y_pred = np.vstack(year_df['Pred'].values)
            
            # Compute the ROC curve for the given diagnosis group and follow-up year.
            roc_curves['ROC_'+dx_bl_name+'_'+str(int(year))] = [get_roc_curve(y_true, y_pred, dx_bl)]
            
    return roc_curves

