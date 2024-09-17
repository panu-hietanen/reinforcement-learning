import matplotlib.pyplot as plt
import numpy as np
import torch

def mean_absolute_diff(y_pred: torch.Tensor, y_pred_noisy: torch.Tensor) -> torch.Tensor:
    """Calculate mean absolute difference between y_pred (single column) and y_pred_noisy (multiple columns)."""
    if y_pred.shape[0] != y_pred_noisy.shape[0]:
        raise ValueError('Incompatible arrays')
    
    # Ensure y_pred is a 2D tensor with shape (num_samples, 1) for broadcasting
    if y_pred.dim() == 1:
        y_pred = y_pred.unsqueeze(1)
    
    # Compute the absolute difference between the single-column y_pred and each column of y_pred_noisy
    abs_diff = torch.abs(y_pred - y_pred_noisy)
    
    # Take the mean along the rows (dim=0) for each column of y_pred_noisy
    mean_diff = torch.mean(abs_diff, dim=0)
    
    return mean_diff


def mean_rmse_change(y_true: torch.Tensor, y_pred: torch.Tensor, y_pred_noisy: torch.Tensor) -> torch.Tensor:
    """Calculate the percentage change in RMSE between original and noisy predictions."""
    
    # Ensure y_true is a column vector for broadcasting
    if y_true.dim() == 1:
        y_true = y_true.unsqueeze(1)
    
    # Compute RMSE for each column of y_pred
    rmse_original = torch.sqrt(torch.mean((y_pred - y_true) ** 2, dim=0))  # Shape: (m,)
    
    # Compute RMSE for each column of y_pred_noisy
    rmse_noisy = torch.sqrt(torch.mean((y_pred_noisy - y_true) ** 2, dim=0))  # Shape: (m,)
    
    # Calculate percentage change in RMSE
    rmse_change = (rmse_noisy - rmse_original) / rmse_original * 100  # Shape: (m,)
    
    return rmse_change



def plot_noise_diff(td_sgd: torch.Tensor, td_adam: torch.Tensor, nn_sgd: torch.Tensor, nn_adam: torch.Tensor) -> None:
    """Plot a bar chart of mean absolute difference."""
    methods = ['TD-SGD', 'TD-Adam', 'L2-SGD', 'L2-Adam']
    changes = [torch.mean(td_sgd), torch.mean(td_adam), torch.mean(nn_sgd), torch.mean(nn_adam)]
    stes = [
        torch.std(td_sgd) /  np.sqrt(len(td_sgd)), 
        torch.std(td_adam) / np.sqrt(len(td_adam)), 
        torch.std(nn_sgd) /  np.sqrt(len(nn_sgd)), 
        torch.std(nn_adam) / np.sqrt(len(nn_adam))
    ]

    # Plot the bar chart with error bars (std deviation)
    plt.bar(methods, changes, yerr=stes, color=['blue', 'green', 'red', 'orange'], capsize=5)
    plt.ylabel('Average Change in Output')
    plt.title('Comparison of Sensitivity of Regression Methods (mean difference)')

    # Show the plot
    plt.show()

def plot_rmse_diff(td_sgd: torch.Tensor, td_adam: torch.Tensor, nn_sgd: torch.Tensor, nn_adam: torch.Tensor) -> None:
    """Plot a bar chart of mean rmse change."""
    methods = ['TD-SGD', 'TD-Adam', 'L2-SGD', 'L2-Adam']
    changes = [torch.mean(td_sgd), torch.mean(td_adam), torch.mean(nn_sgd), torch.mean(nn_adam)]
    stes = [
        torch.std(td_sgd) /  np.sqrt(len(td_sgd)), 
        torch.std(td_adam) / np.sqrt(len(td_adam)), 
        torch.std(nn_sgd) /  np.sqrt(len(nn_sgd)), 
        torch.std(nn_adam) / np.sqrt(len(nn_adam))
    ]

    # Plot the bar chart with error bars (std deviation)
    plt.bar(methods, changes, yerr=stes, color=['blue', 'green', 'red', 'orange'], capsize=5)
    plt.ylabel('Average Change in rmse/%')
    plt.title('Comparison of Sensitivity of Regression Methods (rmse change)')

    # Show the plot
    plt.show()