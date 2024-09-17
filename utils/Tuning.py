import torch
import random
from typing import Any, Callable

# Import the new classes
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

# Define the function to perform random search
def random_search(
        model_type: str,
        X_train: torch.Tensor, 
        y_train: torch.Tensor, 
        X_val: torch.Tensor, 
        y_val: torch.Tensor, 
        param_grid: dict[str, Any], 
        n_iter_search: int = 10, 
        optimizer: str = None,
        random_state: int = None
    ) -> dict[str, Any]:
    best_loss = float('inf')
    best_params = None

    if model_type == 'nn':
        for i in range(n_iter_search):
            # Randomly sample a set of hyperparameters
            try:
                optimizer_type = optimizer or random.choice(['sgd', 'adam'])
                learning_rate = random.choice(param_grid['learning_rate'])
                epochs = random.choice(param_grid['epochs'])
                betas = random.choice(param_grid.get('betas', [(0.9, 0.999)]))
            except KeyError:
                print('Please provide a correctly formatted parameter grid.')
                return None

            print(f"Iteration {i+1}: Training NN with optimizer={optimizer_type}, "
                  f"learning_rate={learning_rate}, epochs={epochs}, betas={betas}")

            # Create and evaluate the model with the sampled hyperparameters
            model = create_model_nn(
                optimizer_type=optimizer_type,
                input_size=X_train.shape[1],
                output_size=1,
                learning_rate=learning_rate,
                betas=betas,
            )

            model.fit(X_train, y_train, epochs=epochs)
            loss = model.rmse(X_val, y_val)

            print(f"Validation RMSE: {loss:.4f}")

            # Track the best set of hyperparameters
            if loss < best_loss:
                best_loss = loss
                best_params = {
                    'optimizer_type': optimizer_type,
                    'learning_rate': learning_rate,
                    'epochs': epochs,
                    'betas': betas
                }

        print(f"Best hyperparameters: {best_params}, Best RMSE: {best_loss:.4f}")
        return best_params

    elif model_type == 'td':
        for i in range(n_iter_search):
            # Randomly sample a set of hyperparameters
            try:
                optimizer_type = optimizer or random.choice(['sgd', 'adam'])
                learning_rate = random.choice(param_grid['learning_rate'])
                gamma = random.choice(param_grid['gamma'])
                epsilon = random.choice(param_grid['epsilon'])
                epochs = random.choice(param_grid['epochs'])
                betas = random.choice(param_grid.get('betas', [(0.9, 0.999)]))
            except KeyError as e:
                print('Please provide a correctly formatted parameter grid. The following is not included:')
                print(e)
                return None

            print(f"Iteration {i+1}: Training TD with optimizer={optimizer_type}, "
                  f"learning_rate={learning_rate}, gamma={gamma}, epsilon={epsilon}, "
                  f"epochs={epochs}, betas={betas}")

            # Create the transition matrix P (example with uniform probability)
            num_samples = X_train.shape[0]
            P = torch.ones((num_samples, num_samples)) / num_samples  # Equal probability for each state

            # Create and evaluate the TD model
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

            model.fit(X_train, y_train, epochs=epochs)
            loss = model.rmse(X_val, y_val)

            print(f"Validation RMSE for TD: {loss:.4f}")

            # Track the best set of hyperparameters
            if loss < best_loss:
                best_loss = loss
                best_params = {
                    'optimizer_type': optimizer_type,
                    'learning_rate': learning_rate,
                    'gamma': gamma,
                    'epsilon': epsilon,
                    'epochs': epochs,
                    'betas': betas
                }

        print(f"Best hyperparameters for TD: {best_params}, Best RMSE: {best_loss:.4f}")
        return best_params

    else:
        raise ValueError("Please set model_type to 'nn' or 'td'.")
