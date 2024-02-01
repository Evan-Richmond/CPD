import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from tqdm import tqdm_notebook
from copy import deepcopy


from classifiers import FitMyNN
from algorithms import ChangePointDetectionClassifier_RuLSIF

from metrics import change_point_roc_auc

def generate_synthetic_data_v2(N, change_points_info, rv_params):
    """
    Generate a synthetic dataset with manually specified change points, supporting different distributions and parameters.
    
    Parameters:
    - N (int): Total length of the dataset.
    - change_points_info (dict): A dictionary specifying 'indices' (list of change point indices) and 'count' (number of change points).
    - rv_params (list): A nested list where each sub-list corresponds to a random variable (r.v.) and contains
                        dictionaries with keys 'dist' (distribution name) and distribution parameters for each segment.
    
    Returns:
    - np.ndarray: A 2D numpy array where each row corresponds to an r.v. and each column to a data point.
    """
    # Validate the change point information
    if change_points_info['count'] != len(change_points_info['indices']):
        raise ValueError("The count of change points does not match the number of indices provided.")
    
    # Initialize the dataset array
    data = np.zeros((len(rv_params), N))
    
    # Ensure the list of change point indices starts with 0 and ends with N for iteration
    cps = [0] + change_points_info['indices'] + [N]
    
    # Iterate over each segment defined by consecutive change points
    for i in range(len(cps) - 1):
        start, end = cps[i], cps[i+1]  # Start and end points of the current segment
        
        # Iterate over each random variable
        for rv_index, rv in enumerate(rv_params):
            # Get the parameters for the current r.v. and segment
            segment_params = rv[i]
            
            if segment_params['dist'] == 'normal':
                data[rv_index, start:end] = np.random.normal(segment_params['mu'], segment_params['sigma'], end - start)
            elif segment_params['dist'] == 'uniform':
                data[rv_index, start:end] = np.random.uniform(segment_params['low'], segment_params['high'], end - start)
            # Add more distribution cases as needed
            
    return data

N = 1000  # Total number of data points
dims = 5  # Number of dimensions
change_points_info = {'indices': [300, 600], 'count': 2}  # Change point information
rv_params = [
    [  
        {'dist': 'normal', 'mu': 0, 'sigma': 1},
        {'dist': 'uniform', 'low': 0, 'high': 5},
        {'dist': 'normal', 'mu': 5, 'sigma': 2},
    ],
    [  
        {'dist': 'uniform', 'low': -5, 'high': 0},
        {'dist': 'normal', 'mu': 2, 'sigma': 1},
        {'dist': 'uniform', 'low': 5, 'high': 10},
    ]
]
# Generate the synthetic dataset
synthetic_data = generate_synthetic_data_v2(N, change_points_info, rv_params)

print("Data generated successfully") # Testing

plot_synthetic_data_separate(synthetic_data_5d, change_points_5d)

# Prepare the data (you might need to adjust this based on your data generation function's output)
X_full = synthetic_data.T  # Assuming synthetic_data shape is (dims, N)

# Split the data into training and testing
n_train = int(0.7 * X_full.shape[0])
X_train = X_full[:n_train]
y_train = np.zeros(X_train.shape[0])
for cp in change_points_info['indices']:
    if cp < n_train:
        y_train[cp:] = 1  # Label data after each change point as 1

# Instantiate and train the classifier
classifier = FitMyNN(n_hidden=20, dropout=0.5, n_epochs=50)
classifier.fit(X_train, y_train)

# Prepare test data
X_test = X_full[n_train:]

# Instantiate and run change point detection
cpd = ChangePointDetectionClassifier_RuLSIF(base_classifier=classifier, window_size=50, step=10)
T_scores, scores = cpd.predict(X_test)

cpd = ChangePointDetectionClassifier_RuLSIF(base_classifier=classifier, window_size=50, step=10)

# Prepare true labels for test data
y_test = np.zeros(X_test.shape[0])
for cp in change_points_info['indices']:
    if n_train <= cp < N:
        y_test[cp - n_train:] = 1

# Evaluate change point detection
auc = change_point_roc_auc(np.arange(len(y_test)), y_test, T_scores, scores, window_size=50)

print(f"ROC AUC Score: {auc}")
