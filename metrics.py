import numpy as np
import pandas as pd

from sklearn.metrics import roc_curve, roc_auc_score

def change_point_roc_auc(T_label, label, T_score, score, window_size):
    """
    T_label : numpy.array
        Ineces of time series observations and their labels.
    label   : numpy.array
        Labels of time series observations. 1 - change point, 0 - ordinary observation.
    T_score : numpy.array
        Indeces of change point detection score.
    score   : numpy.array
        Change point detection score. 0 - ordinary observation, high value - change point.
    window_size : int
        All observations with t < 2 * window_size after a change point are marked as 1 
        and considered as a new collective change point.
        
    Returns
    -------
    auc : float
        ROC AUC score calculated based on change point detection score and new collective change points.
    """
    
    new_label = np.zeros(len(label))
    T_change = T_label[label == 1]
    for t in T_change:
        cond = (T_label-t < 2 * window_size) * (T_label-t >= 0)
        new_label[cond] = 1
        
    new_label = new_label[T_score]
    auc = roc_auc_score(new_label, score)
    
    return auc