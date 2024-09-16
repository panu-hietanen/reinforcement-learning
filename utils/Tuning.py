import torch
import random
from typing import Any

from utils.NeuralNet import BaseThreeLayerFCNN, ThreeLayerFCNN_Adam, ThreeLayerFCNN_SGD
from utils.TD import BaseTD, TD_Adam, TD_SGD
from utils.TD_NN import BaseTD_NN, TD_NN_Adam, TD_NN_SGD

# Assuming NN_SGD or NN_Adam are your neural network classes
# Define a function to create the model
def create_model_nn(optimizer_type: str, input_size: int, hidden_size: int, lr: float, batch_size: int, n_epochs: int, betas: tuple[float, float] = (0.9, 0.999)) -> BaseThreeLayerFCNN:
    if optimizer_type == 'sgd':
        return ThreeLayerFCNN_SGD(batch_size=batch_size, lr=lr, n_epochs=n_epochs, input_size=input_size, hidden_size=hidden_size)
    elif optimizer_type == 'adam':
        return ThreeLayerFCNN_Adam(batch_size=batch_size, lr=lr, n_epochs=n_epochs, input_size=input_size, hidden_size=hidden_size, betas=betas)
    else:
        raise ValueError("Invalid optimizer type")
    
# Function to create TD model
def create_model_td(optimizer_type: str, n_iter: int, gamma: float, alpha: float, epsilon: float, P: torch.Tensor, betas: tuple[float, float] = (0.9, 0.999)) -> BaseTD:
    if optimizer_type == 'sgd':
        return TD_SGD(n_iter=n_iter, P=P, link=lambda x: x, inv_link=lambda x: x, gamma=gamma, alpha=alpha, epsilon=epsilon)
    elif optimizer_type == 'adam':
        return TD_Adam(n_iter=n_iter, P=P, link=lambda x: x, inv_link=lambda x: x, gamma=gamma, alpha=alpha, epsilon=epsilon, betas=betas)
    else:
        raise ValueError(f"Invalid optimizer type: {optimizer_type}")
    
def create_model_td_nn(optimizer_type: str, n_iter: int, input_size: int, gamma: float, alpha: float, epsilon: float, P: torch.Tensor, betas: tuple[float, float] = (0.9, 0.999)) -> BaseTD_NN:
    if optimizer_type == 'sgd':
        return TD_NN_SGD(n_iter=n_iter, P=P, link=lambda x: x, inv_link=lambda x: x, gamma=gamma, alpha=alpha, epsilon=epsilon, input_size=input_size)
    elif optimizer_type == 'adam':
        return TD_NN_Adam(n_iter=n_iter, P=P, link=lambda x: x, inv_link=lambda x: x, gamma=gamma, alpha=alpha, epsilon=epsilon, input_size=input_size, betas=betas)
    else:
        raise ValueError(f"Invalid optimizer type: {optimizer_type}")


# Define the function to perform random search
def random_search(
        model_type: str,
        X_train: torch.Tensor, 
        y_train: torch.Tensor, 
        X_val: torch.Tensor, 
        y_val: torch.Tensor, 
        param_grid: dict[str, Any], 
        n_iter: int = 10, 
        optimizer: str = None,
        ) -> dict[str, Any]:
    best_loss = float('inf')
    best_params = None

    if model_type == 'nn':
        for i in range(n_iter):
            # Randomly sample a set of hyperparameters
            try:
                optimizer_type = (optimizer or random.choice(['sgd', 'adam']))
                hidden_size = random.choice(param_grid['hidden_size'])
                lr = random.choice(param_grid['lr'])
                batch_size = random.choice(param_grid['batch_size'])
                n_epochs = random.choice(param_grid['n_epochs'])
            except:
                print('Please enter the correctly formatted dictionary.')
                return None
            
            if 'betas' in param_grid:
                betas = random.choice(param_grid['betas'])
            else:
                betas = (0.9, 0.999)
            
            if optimizer_type == 'adam':
                print(f"Iteration {i+1}: Training NN with optimizer={optimizer_type}, hidden_size={hidden_size}, "
                      f"lr={lr}, batch_size={batch_size}, n_epochs={n_epochs}, betas={betas}")
            else:
                print(f"Iteration {i+1}: Training NN with optimizer={optimizer_type}, hidden_size={hidden_size}, "
                      f"lr={lr}, batch_size={batch_size}, n_epochs={n_epochs}")

            # Create and evaluate the model with the sampled hyperparameters
            model = create_model_nn(optimizer_type, X_train.shape[1], hidden_size, lr, batch_size, n_epochs, betas=betas)
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
                    'n_epochs': n_epochs,
                    'betas': betas
                }

        print(f"Best hyperparameters: {best_params}, Best RMSE: {best_loss}")
        return best_params
    elif model_type == 'td':
        for i in range(n_iter):
            # Randomly sample a set of hyperparameters
            try:
                optimizer_type = (optimizer or random.choice(['sgd', 'adam']))
                n_iter_td = random.choice(param_grid['n_iter'])
                gamma = random.choice(param_grid['gamma'])
                alpha = random.choice(param_grid['alpha'])
                epsilon = random.choice(param_grid['epsilon'])
            except KeyError as e:
                print('Please enter the correctly formatted dictionary.')
                return None
            
            if 'betas' in param_grid:
                betas = random.choice(param_grid['betas'])
            else:
                betas = (0.9, 0.999)

            if optimizer_type == 'adam':
                print(f"Iteration {i+1}: Training TD with optimizer={optimizer_type}, n_iter={n_iter_td}, "
                      f"gamma={gamma}, alpha={alpha}, epsilon={epsilon}, betas={betas}")
            else:
                print(f"Iteration {i+1}: Training TD with optimizer={optimizer_type}, n_iter={n_iter_td}, "
                      f"gamma={gamma}, alpha={alpha}, epsilon={epsilon}")

            # Create the transition matrix P (example with uniform probability)
            num_samples = X_train.shape[0]
            P = torch.ones((num_samples, num_samples)) / num_samples  # Equal probability for each state

            # Create and evaluate the TD model
            model = create_model_td(optimizer_type, n_iter_td, gamma, alpha, epsilon, P, betas=betas)
            model.fit(X_train, y_train)
            loss = model.rmse(X_val, y_val)

            print(f"Validation RMSE for TD: {loss}")

            # Track the best set of hyperparameters
            if loss < best_loss:
                best_loss = loss
                best_params = {
                    'optimizer_type': optimizer_type,
                    'n_iter': n_iter_td,
                    'gamma': gamma,
                    'alpha': alpha,
                    'epsilon': epsilon,
                    'betas': betas
                }

        print(f"Best hyperparameters for TD: {best_params}, Best RMSE: {best_loss}")
        return best_params
    elif model_type == 'td_nn':
        for i in range(n_iter):
            # Randomly sample a set of hyperparameters
            try:
                optimizer_type = (optimizer or random.choice(['sgd', 'adam']))
                n_iter_td = random.choice(param_grid['n_iter'])
                gamma = random.choice(param_grid['gamma'])
                alpha = random.choice(param_grid['alpha'])
                epsilon = random.choice(param_grid['epsilon'])
            except KeyError as e:
                print('Please enter the correctly formatted dictionary.')
                return None
            
            if 'betas' in param_grid:
                betas = random.choice(param_grid['betas'])
            else:
                betas = (0.9, 0.999)

            if optimizer_type == 'adam':
                print(f"Iteration {i+1}: Training TD with optimizer={optimizer_type}, n_iter={n_iter_td}, "
                      f"gamma={gamma}, alpha={alpha}, epsilon={epsilon}, betas={betas}")
            else:
                print(f"Iteration {i+1}: Training TD with optimizer={optimizer_type}, n_iter={n_iter_td}, "
                      f"gamma={gamma}, alpha={alpha}, epsilon={epsilon}")

            # Create the transition matrix P (example with uniform probability)
            num_samples = X_train.shape[0]
            P = torch.ones((num_samples, num_samples)) / num_samples  # Equal probability for each state

            # Create and evaluate the TD model
            model = create_model_td_nn(optimizer_type, n_iter_td, X_train.shape[1], gamma, alpha, epsilon, P, betas=betas)
            model.fit(X_train, y_train)
            loss = model.rmse(X_val, y_val)

            print(f"Validation RMSE for TD: {loss}")

            # Track the best set of hyperparameters
            if loss < best_loss:
                best_loss = loss
                best_params = {
                    'optimizer_type': optimizer_type,
                    'n_iter': n_iter_td,
                    'gamma': gamma,
                    'alpha': alpha,
                    'epsilon': epsilon,
                    'betas': betas
                }

        print(f"Best hyperparameters for TD: {best_params}, Best RMSE: {best_loss}")
        return best_params
    else:
        raise ValueError("Please either set the model as td or nn.")
