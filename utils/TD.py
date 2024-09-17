"""
Code containing a class for performing standardised Temporal Difference learning.

Uses methods from https://arxiv.org/abs/2404.15518v3

Date: 10/9/24
Author: Panu Hietanen
"""
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Callable

class TemporalDifferenceLinear:
    def __init__(
            self,
            optimizer: str,
            input_size: int,
            output_size: int = 1,
            learning_rate: float = 0.01,
            gamma: float = 0,
            epsilon: float = 0,
            n_iter: int = 1000,
            P: torch.Tensor = None,
            link: Callable[[torch.Tensor], torch.Tensor] = None,
            inv_link: Callable[[torch.Tensor], torch.Tensor] = None,
            betas: tuple[float, float] = (0.9, 0.999),
            random_state: int = None,
        ) -> None:
        # Set seed for reproducibility
        if random_state is not None:
            torch.manual_seed(random_state)

        # Initialize linear model
        self.model = nn.Linear(input_size, output_size)

        # Select optimizer
        if optimizer == 'sgd':
            self.optimizer = optim.SGD(self.model.parameters(), lr=learning_rate)
        elif optimizer == 'adam':
            self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, betas=betas)
        else:
            raise ValueError("Optimizer must be 'sgd' or 'adam'")

        # Mean Squared Error loss
        self.criterion = nn.MSELoss()

        # TD Learning parameters
        self.gamma = gamma
        self.epsilon = epsilon
        self.n_iter = int(n_iter)
        self.P = P
        self.link = link if link else lambda x: x  # Identity function if None
        self.inv_link = inv_link if inv_link else lambda x: x  # Identity function if None

        # Validate the transition matrix P
        if self.P is None:
            raise ValueError("Transition matrix P must be provided.")
        if not torch.allclose(self.P.sum(dim=1), torch.ones(self.P.size(0))):
            raise ValueError("Each row of the transition matrix P must sum to 1.")

        self.trained = False
        self.rng = torch.Generator()
        if random_state is not None:
            self.rng.manual_seed(random_state)

    def sample_next_state(self, index: int) -> int:
        """Sample the next state based on the transition matrix P."""
        probs = self.P[index]
        next_state = torch.multinomial(probs, 1, generator=self.rng)
        return next_state.item()

    def fit(self, X: torch.Tensor, y: torch.Tensor) -> None:
        n_samples, n_features = X.shape

        if y.size(0) != n_samples:
            raise ValueError("Ensure there are the same number of target samples as feature samples.")

        # Ensure inputs are tensors
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, dtype=torch.float32)
        if not isinstance(y, torch.Tensor):
            y = torch.tensor(y, dtype=torch.float32)

        # Set the model to training mode
        self.model.train()

        curr_index = torch.randint(0, n_samples, (1,), generator=self.rng).item()
        curr_x = X[curr_index].unsqueeze(0)  # Add batch dimension
        curr_y = y[curr_index]

        for i in range(self.n_iter):
            # Next state samples
            next_index = self.sample_next_state(curr_index)
            next_x = X[next_index].unsqueeze(0)  # Add batch dimension
            next_y = y[next_index]

            # Zero the gradients
            self.optimizer.zero_grad()

            # Forward pass for current and next states
            curr_z = self.model(curr_x).squeeze()
            with torch.no_grad():
                next_z = self.model(next_x).squeeze()

            # Compute reward
            r = self.inv_link(curr_y) - self.gamma * self.inv_link(next_y)

            # TD target
            z_t = r + self.gamma * next_z

            # Compute loss
            loss = self.criterion(self.link(curr_z), self.link(z_t))

            # Backward pass and optimization
            loss.backward()
            self.optimizer.step()

            # Early stopping based on loss
            if loss.item() < self.epsilon:
                print(f'Ending optimization early at iteration {i+1}')
                break

            # Update current state
            curr_index = next_index
            curr_x = next_x
            curr_y = next_y

        self.trained = True

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        """Predict outputs for the given input X."""
        if not self.trained:
            raise TDError('Model has not been trained.')
        # Ensure input is a tensor
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, dtype=torch.float32)
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X)
        return outputs.squeeze()

    def rmse(self, X: torch.Tensor, y: torch.Tensor) -> float:
        """Calculate the RMSE between the model predictions and targets."""
        if not self.trained:
            raise TDError('Model has not been trained.')
        # Ensure inputs are tensors
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, dtype=torch.float32)
        if not isinstance(y, torch.Tensor):
            y = torch.tensor(y, dtype=torch.float32)
        with torch.no_grad():
            outputs = self.predict(X)
            mse_loss = self.criterion(outputs, y)
            rmse = torch.sqrt(mse_loss)
        return rmse.item()

    def reset(self) -> None:
        """Reset model weights."""
        if not self.trained:
            return
        for layer in self.model.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
        self.trained = False

class TDError(Exception):
    pass
