import matplotlib.pyplot as plt
import numpy as np
import torch

def mean_absolute_diff(y_pred: torch.Tensor, y_pred_noisy: torch.Tensor) -> float:
    """Calculate mean absolute difference between y and yhat."""
    return torch.mean(torch.abs(y_pred - y_pred_noisy)).item()


def plot_noise_diff(td_sgd: float, td_adam: float, nn_sgd: float, nn_adam: float) -> None:
    """Plot a bar chart of mean absolute difference."""
    methods = ['TD-SGD', 'TD-Adam', 'NN-SGD', 'NN-Adam']
    changes = [td_sgd, td_adam, nn_sgd, nn_adam]

    plt.bar(methods, changes, color=['blue', 'green', 'red', 'orange'])
    plt.ylabel('Average Change in Output')
    plt.title('Comparison of Regression Methods with Gaussian Noise')
    plt.show()