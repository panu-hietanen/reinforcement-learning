import torch
import random
from typing import Any, Callable, List, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed
from scipy.stats.qmc import Halton
from utils.TD_NN import TemporalDifferenceNN
from utils.NeuralNet import TwoHiddenLayerNN


# Function to create the neural network model
def create_model_nn(
        optimizer_type: str,
        input_size: int,
        output_size: int,
        learning_rate: float,
        betas: tuple[float, float] = (0.9, 0.999),
    ) -> TwoHiddenLayerNN:
    return TwoHiddenLayerNN(
        input_size=input_size,
        output_size=output_size,
        optimizer=optimizer_type,
        learning_rate=learning_rate,
        betas=betas,
    )

# Function to create the TD model
def create_model_td(
        optimizer_type: str,
        input_size: int,
        learning_rate: float,
        gamma: float,
        epsilon: float,
        P: torch.Tensor,
        betas: tuple[float, float] = (0.9, 0.999),
        random_state: int = None,
        link: Callable[[torch.Tensor], torch.Tensor] = None,
        inv_link: Callable[[torch.Tensor], torch.Tensor] = None
    ) -> TemporalDifferenceNN:
    return TemporalDifferenceNN(
        input_size=input_size,
        optimizer=optimizer_type,
        learning_rate=learning_rate,
        gamma=gamma,
        epsilon=epsilon,
        P=P,
        betas=betas,
        random_state=random_state,
        link=link,
        inv_link=inv_link
    )

# Function to perform random search for the neural network model
def random_search_nn(
        X_train: torch.Tensor, 
        y_train: torch.Tensor, 
        X_val: torch.Tensor, 
        y_val: torch.Tensor, 
        param_grid: Dict[str, List[Any]], 
        n_iter_search: int = 10, 
        optimizer: str = None,
        random_state: int = None,
        max_workers: int = 4
    ) -> Dict[str, Any]:
    best_loss = float('inf')
    best_params = None

    # Set up Halton sequence sampler
    sampler = Halton(d=len(param_grid), scramble=True, seed=random_state)
    sample_points = sampler.random(n_iter_search)
    
    # Map parameter names to indices
    param_names = list(param_grid.keys())
    param_indices = {name: idx for idx, name in enumerate(param_names)}

    futures = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for i in range(n_iter_search):
            # Sample hyperparameters using Halton sequence
            hyperparams = {}
            for param_name in param_names:
                values = param_grid[param_name]
                idx = param_indices[param_name]
                value_idx = int(sample_points[i][idx] * len(values))
                value_idx = min(value_idx, len(values) - 1)  # Ensure index is within bounds
                hyperparams[param_name] = values[value_idx]

            # Prepare arguments for training
            optimizer_type = optimizer or hyperparams.get('optimizer', random.choice(['sgd', 'adam']))
            learning_rate = hyperparams['learning_rate']
            epochs = hyperparams['epochs']
            betas = hyperparams.get('betas', (0.9, 0.999))

            # Submit training to executor
            futures.append(executor.submit(
                train_and_evaluate_nn,
                X_train, y_train, X_val, y_val,
                optimizer_type, learning_rate, epochs, betas,
                random_state + i if random_state is not None else None
            ))

    for future in as_completed(futures):
        loss, params = future.result()
        print(f"Validation RMSE: {loss:.4f} for params: {params}")

        # Track the best set of hyperparameters
        if loss < best_loss:
            best_loss = loss
            best_params = params

    print(f"Best hyperparameters: {best_params}, Best RMSE: {best_loss:.4f}")
    return best_params

def train_and_evaluate_nn(
        X_train: torch.Tensor, 
        y_train: torch.Tensor, 
        X_val: torch.Tensor, 
        y_val: torch.Tensor,
        optimizer_type: str, 
        learning_rate: float, 
        epochs: int, 
        betas: tuple[float, float],
        random_state: int,
    ) -> tuple[float, Dict[str, Any]]:
    # Set random seed for reproducibility
    if random_state is not None:
        torch.manual_seed(random_state)
        random.seed(random_state)
    
    # Create and train the model
    model = create_model_nn(
        optimizer_type=optimizer_type,
        input_size=X_train.shape[1],
        output_size=1,
        learning_rate=learning_rate,
        betas=betas,
    )

    model.fit(X_train, y_train, epochs=epochs)  # Pass validation data for early stopping
    loss = model.rmse(X_val, y_val)

    params = {
        'optimizer_type': optimizer_type,
        'learning_rate': learning_rate,
        'epochs': epochs,
        'betas': betas
    }
    return loss, params


# Function to perform random search for the TD model
def random_search_td(
        X_train: torch.Tensor, 
        y_train: torch.Tensor, 
        X_val: torch.Tensor, 
        y_val: torch.Tensor, 
        param_grid: Dict[str, List[Any]], 
        n_iter_search: int = 10, 
        optimizer: str = None,
        random_state: int = None,
        max_workers: int = 4
    ) -> Dict[str, Any]:
    best_loss = float('inf')
    best_params = None

    # Set up Halton sequence sampler
    sampler = Halton(d=len(param_grid), scramble=True, seed=random_state)
    sample_points = sampler.random(n_iter_search)
    
    # Map parameter names to indices
    param_names = list(param_grid.keys())
    param_indices = {name: idx for idx, name in enumerate(param_names)}

    futures = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for i in range(n_iter_search):
            # Sample hyperparameters using Halton sequence
            hyperparams = {}
            for param_name in param_names:
                values = param_grid[param_name]
                idx = param_indices[param_name]
                value_idx = int(sample_points[i][idx] * len(values))
                value_idx = min(value_idx, len(values) - 1)  # Ensure index is within bounds
                hyperparams[param_name] = values[value_idx]

            # Prepare arguments for training
            optimizer_type = optimizer or hyperparams.get('optimizer', random.choice(['sgd', 'adam']))
            learning_rate = hyperparams['learning_rate']
            gamma = hyperparams['gamma']
            epsilon = hyperparams['epsilon']
            epochs = hyperparams['epochs']
            betas = hyperparams.get('betas', (0.9, 0.999))

            # Create the transition matrix P (example with uniform probability)
            num_samples = X_train.shape[0]
            P = torch.ones((num_samples, num_samples)) / num_samples  # Equal probability for each state

            # Submit training to executor
            futures.append(executor.submit(
                train_and_evaluate_td,
                X_train, y_train, X_val, y_val,
                optimizer_type, learning_rate, gamma, epsilon, epochs, betas, P,
                random_state + i if random_state is not None else None
            ))

    for future in as_completed(futures):
        loss, params = future.result()
        print(f"Validation RMSE for TD: {loss:.4f} for params: {params}")

        # Track the best set of hyperparameters
        if loss < best_loss:
            best_loss = loss
            best_params = params

    print(f"Best hyperparameters for TD: {best_params}, Best RMSE: {best_loss:.4f}")
    return best_params

def train_and_evaluate_td(
        X_train: torch.Tensor, 
        y_train: torch.Tensor, 
        X_val: torch.Tensor, 
        y_val: torch.Tensor,
        optimizer_type: str, 
        learning_rate: float, 
        gamma: float, 
        epsilon: float, 
        epochs: int,
        betas: tuple[float, float], 
        P: torch.Tensor,
        random_state: int,
    ) -> tuple[float, Dict[str, Any]]:
    # Set random seed for reproducibility
    if random_state is not None:
        torch.manual_seed(random_state)
        random.seed(random_state)
    
    # Create and train the model
    model = create_model_td(
        optimizer_type=optimizer_type,
        input_size=X_train.shape[1],
        learning_rate=learning_rate,
        gamma=gamma,
        epsilon=epsilon,
        P=P,
        betas=betas,
        random_state=random_state,
        link=lambda x: x,  # Identity link function
        inv_link=lambda x: x  # Identity inverse link function
    )

    model.fit(X_train, y_train, epochs=epochs)  # Pass validation data for early stopping
    loss = model.rmse(X_val, y_val)

    params = {
        'optimizer_type': optimizer_type,
        'learning_rate': learning_rate,
        'gamma': gamma,
        'epsilon': epsilon,
        'epochs': epochs,
        'betas': betas
    }
    return loss, params


