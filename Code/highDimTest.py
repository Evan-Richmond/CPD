import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm  
from copy import deepcopy

# Assuming classifiers.py and algorithms.py are in the same directory
from classifiers import FitMyNN
from algorithms import ChangePointDetectionClassifier_RuLSIF

from sklearn.metrics import roc_curve, roc_auc_score

# Modified roc_auc
def change_point_roc_auc(T_label, label, T_score, score, window_size):
    new_label = np.zeros(len(label))
    T_change = T_label[label == 1]
    for t in T_change:
        cond = (T_label-t < 2 * window_size) & (T_label-t >= 0)
        new_label[cond] = 1
        
    new_label = new_label[T_score]
    
    # Check if new_label has both classes present
    if len(np.unique(new_label)) < 2:
        # Handle the single-class case appropriately
        print("Warning: Only one class present in y_true. ROC AUC score is not defined in that case.")
        return None  # Or return an indicative value like 0.5 or a message
    
    auc = roc_auc_score(new_label, score)
    return auc

# High-dimensional synthetic data generation function
def generate_synthetic_data_v2(N, change_points_info, rv_params):
    if change_points_info['count'] != len(change_points_info['indices']):
        raise ValueError("The count of change points does not match the number of indices provided.")
    data = np.zeros((len(rv_params), N))
    cps = [0] + change_points_info['indices'] + [N]
    for i in range(len(cps) - 1):
        start, end = cps[i], cps[i + 1]
        for rv_index, rv in enumerate(rv_params):
            segment_params = rv[i]
            if segment_params['dist'] == 'normal':
                data[rv_index, start:end] = np.random.normal(segment_params['mu'], segment_params['sigma'], end - start)
            elif segment_params['dist'] == 'uniform':
                data[rv_index, start:end] = np.random.uniform(segment_params['low'], segment_params['high'], end - start)
    return data

def plot_synthetic_data_separate(data, change_points_info):
    """
    Plot each dimension of the synthetic data on a separate subplot.
    
    Parameters:
    - data (np.ndarray): The synthetic dataset to plot, where each row is a random variable.
    - change_points_info (dict): Information about the change points to highlight them in the plot.
    """
    num_variables = data.shape[0]  # Number of dimensions/random variables
    fig, axes = plt.subplots(num_variables, 1, figsize=(12, 2 * num_variables))  # Create subplots for each dimension
    
    for i, ax in enumerate(axes):
        ax.plot(data[i, :], label=f'Dimension {i+1}')
        
        # Highlight change points
        for cp in change_points_info['indices']:
            ax.axvline(x=cp, linestyle='--', color='gray', alpha=0.7)
        
        ax.set_title(f'Dimension {i+1}')
        ax.set_xlabel('Data Point Index')
        ax.set_ylabel('Value')
        ax.legend()
    
    plt.tight_layout()
    plt.show()

# Data generation parameters
N = 1000
dims = 5
change_points_info = {'indices': [300, 600], 'count': 2}
rv_params = [
    [{'dist': 'normal', 'mu': 0, 'sigma': 1}, {'dist': 'uniform', 'low': 0, 'high': 5}, {'dist': 'normal', 'mu': 5, 'sigma': 2}],
    [{'dist': 'uniform', 'low': -5, 'high': 0}, {'dist': 'normal', 'mu': 2, 'sigma': 1}, {'dist': 'uniform', 'low': 5, 'high': 10}]
]

# Generate synthetic data
synthetic_data = generate_synthetic_data_v2(N, change_points_info, rv_params)

print("Data generated!") # Testing

plot_synthetic_data_separate(synthetic_data, change_points_info) # Plot data

X_full = synthetic_data.T

# Split data into training and testing sets
n_train = int(0.7 * X_full.shape[0])
X_train, X_test = X_full[:n_train], X_full[n_train:]

# Create labels for training data
y_train = np.zeros(n_train)
for cp in change_points_info['indices']:
    if cp < n_train:
        y_train[cp:] = 1

# Train the classifier
classifier = FitMyNN(n_hidden=20, dropout=0.5, n_epochs=50)
classifier.fit(X_train, y_train)

# Detect change points using the trained classifier
cpd = ChangePointDetectionClassifier_RuLSIF(base_classifier=classifier, window_size=50, step=10, n_runs=1, debug=1)
T_scores, scores = cpd.predict(X_test)

# Adjust labeling for test data considering window_size
window_size = 50  # Ensure this matches the window_size used in ChangePointDetectionClassifier_RuLSIF
y_test = np.zeros(len(X_test))
for cp in change_points_info['indices']:
    if n_train <= cp < N:
        start_index = max(cp - n_train - window_size, 0)
        end_index = min(cp - n_train + window_size, len(y_test))
        y_test[start_index:end_index] = 1

# Verify if both classes are present in y_test
if len(np.unique(y_test)) < 2:
    print("Warning: Not enough change points in the test set for ROC AUC calculation.")

# Evaluate change point detection using ROC AUC
auc = change_point_roc_auc(np.arange(len(y_test)), y_test, T_scores, scores, window_size)
print(f"ROC AUC Score: {auc}")

# Improved visualization
plt.figure(figsize=(14, 6))
plt.plot(scores, label='Change Point Score')
plt.scatter(np.arange(len(y_test)), y_test * max(scores), c='red', label='True Change Points', marker='x')
plt.axhline(y=np.mean(scores) + 2*np.std(scores), color='r', linestyle='--', label='Threshold')
plt.legend()
plt.title('Change Point Scores with Threshold and True Change Points')
plt.xlabel('Time')
plt.ylabel('Score')
plt.show()
