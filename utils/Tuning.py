import torch
import random
from typing import Any

from utils.NeuralNet import TwoLayerFCNN_Adam, TwoLayerFCNN_SGD
from utils.TD import TD_Adam, TD_SGD

# Assuming NN_SGD or NN_Adam are your neural network classes
# Define a function to create the model
def create_model_nn(optimizer_type: str, input_size: int, hidden_size: int, lr: float, batch_size: int, n_epochs: int):
    if optimizer_type == 'sgd':
        return TwoLayerFCNN_SGD(batch_size=batch_size, lr=lr, n_epochs=n_epochs, input_size=input_size, hidden_size=hidden_size)
    elif optimizer_type == 'adam':
        return TwoLayerFCNN_Adam(batch_size=batch_size, lr=lr, n_epochs=n_epochs, input_size=input_size, hidden_size=hidden_size)
    else:
        raise ValueError("Invalid optimizer type")

# Define the function to perform random search
def random_search(X_train: torch.Tensor, y_train: torch.Tensor, X_val: torch.Tensor, y_val: torch.Tensor, param_grid: dict[str, Any], n_iter: int = 10, optimizer: str = None):
    best_loss = float('inf')
    best_params = None

    for i in range(n_iter):
        # Randomly sample a set of hyperparameters
        optimizer_type = (optimizer or random.choice(['sgd', 'adam']))
        hidden_size = random.choice(param_grid['hidden_size'])
        lr = random.choice(param_grid['lr'])
        batch_size = random.choice(param_grid['batch_size'])
        n_epochs = random.choice(param_grid['n_epochs'])

        print(f"Iteration {i+1}: Training with optimizer={optimizer_type}, hidden_size={hidden_size}, "
              f"lr={lr}, batch_size={batch_size}, n_epochs={n_epochs}")

        # Create and evaluate the model with the sampled hyperparameters
        model = create_model_nn(optimizer_type, X_train.shape[1], hidden_size, lr, batch_size, n_epochs)
        model.fit(X_train, y_train)
        loss = model.evaluate(X_val, y_val)

        print(f"Validation RMSE: {loss}")

        # Track the best set of hyperparameters
        if loss < best_loss:
            best_loss = loss
            best_params = {
                'optimizer_type': optimizer_type,
                'hidden_size': hidden_size,
                'lr': lr,
                'batch_size': batch_size,
                'n_epochs': n_epochs
            }

    print(f"Best hyperparameters: {best_params}, Best RMSE: {best_loss}")
    return best_params
