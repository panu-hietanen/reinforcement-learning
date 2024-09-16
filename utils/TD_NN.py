"""
Code containing a class for performing Temporal Difference learning using a neural network.

Date: 10/9/24
Author: Panu Hietanen
"""

import torch
from torch import nn, optim
from typing import Callable
from abc import ABC, abstractmethod
from utils.NeuralNet import BaseThreeLayerFCNN

class BaseTD_NN(ABC):
    """Class to perform optimization using generalized TD methods with a neural network."""

    def __init__(
            self,
            n_iter: int,
            P: torch.Tensor,
            link: Callable[[torch.Tensor], torch.Tensor],
            inv_link: Callable[[torch.Tensor], torch.Tensor],
            gamma: float,
            alpha: float,
            epsilon: float,
            input_size: int,
            hidden_size: int = 64,
            output_size: int = 1,
            random_state: int = None,
    ) -> None:
        self.n_iter = n_iter
        self.P = P
        self.link = link
        self.inv_link = inv_link
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Set seed for reproducibility
        self.rng = torch.Generator()
        if random_state is not None:
            self.rng.manual_seed(random_state)

        # Ensure P matrix sums to one (along rows)
        if not torch.allclose(P.sum(dim=1), torch.ones(P.size(0))):
            raise ValueError("Each row of the transition matrix P must sum to 1.")

        # Initialize neural network
        self.model = BaseThreeLayerFCNN(
            batch_size=0,
            lr=self.alpha,
            n_epochs=n_iter,
            input_size=self.input_size
        )

        self.trained = False

    def sample_next_state(self, index: int) -> int:
        """Sample the next state based on the transition matrix P."""
        probs = self.P[index]
        next_state = torch.multinomial(probs, 1, generator=self.rng)
        return next_state.item()

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        if not self.trained:
            raise TDError("Please use the fit function first!")
        with torch.no_grad():
            return self.model(X)

    def rmse(self, X: torch.Tensor, y: torch.Tensor) -> float:
        if not self.trained:
            raise TDError("Please use the fit function first!")
        y_hat = self.predict(X)
        error = y_hat - y.unsqueeze(1)  # Ensure dimensions match
        return torch.sqrt(torch.mean(torch.pow(error, 2)))

    def reset(self) -> None:
        """Reset the model's parameters."""
        def weight_reset(m):
            if isinstance(m, nn.Linear):
                m.reset_parameters()
        self.model.apply(weight_reset)
        self.trained = False

    @abstractmethod
    def fit(self, X: torch.Tensor, y: torch.Tensor) -> None:
        pass

class TD_NN_SGD(BaseTD_NN):
    def fit(self, X: torch.Tensor, y: torch.Tensor) -> None:
        n_samples = X.size(0)

        if y.size(0) != n_samples:
            raise TDError("Ensure there are the same number of target samples as feature samples.")

        optimizer = optim.SGD(self.model.parameters(), lr=self.alpha)
        criterion = nn.MSELoss()

        curr_index = torch.randint(0, n_samples, (1,), generator=self.rng).item()
        curr_x = X[curr_index].unsqueeze(0)  # Add batch dimension
        curr_y = y[curr_index]

        i = 0

        while i < self.n_iter:
            # Next state samples
            next_index = self.sample_next_state(curr_index)
            next_x = X[next_index].unsqueeze(0)  # Add batch dimension
            next_y = y[next_index]

            optimizer.zero_grad()

            # Find predictions
            curr_z = self.model(curr_x)
            next_z = self.model(next_x)

            # Find rewards
            r = self.inv_link(curr_y) - self.gamma * self.inv_link(next_y)

            # TD target
            z_t = r + self.gamma * next_z.detach()

            # Compute loss
            loss = criterion(self.link(curr_z), self.link(z_t))

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            # Update state and index
            curr_index, curr_x, curr_y = next_index, next_x, next_y
            i += 1

            # Early stopping based on loss
            if loss.item() < self.epsilon:
                print(f'Ending optimization early at iteration {i}')
                break

        self.trained = True

class TD_NN_Adam(BaseTD_NN):
    def __init__(
            self,
            n_iter: int,
            P: torch.Tensor,
            link: Callable[[torch.Tensor], torch.Tensor],
            inv_link: Callable[[torch.Tensor], torch.Tensor],
            gamma: float,
            alpha: float,
            epsilon: float,
            input_size: int,
            hidden_size: int = 64,
            output_size: int = 1,
            betas: tuple = (0.9, 0.999),
            random_state: int = None,
    ) -> None:
        super(TD_NN_Adam, self).__init__(n_iter, P, link, inv_link, gamma, alpha, epsilon, input_size, hidden_size, output_size, random_state)
        self.betas = betas

    def fit(self, X: torch.Tensor, y: torch.Tensor) -> None:
        n_samples = X.size(0)

        if y.size(0) != n_samples:
            raise TDError("Ensure there are the same number of target samples as feature samples.")

        optimizer = optim.Adam(self.model.parameters(), lr=self.alpha, betas=self.betas)
        criterion = nn.MSELoss()

        curr_index = torch.randint(0, n_samples, (1,), generator=self.rng).item()
        curr_x = X[curr_index].unsqueeze(0)  # Add batch dimension
        curr_y = y[curr_index]

        i = 0

        while i < self.n_iter:
            # Next state samples
            next_index = self.sample_next_state(curr_index)
            next_x = X[next_index].unsqueeze(0)  # Add batch dimension
            next_y = y[next_index]

            optimizer.zero_grad()

            # Find predictions
            curr_z = self.model(curr_x)
            next_z = self.model(next_x)

            # Find rewards
            r = self.inv_link(curr_y) - self.gamma * self.inv_link(next_y)

            # TD target
            z_t = r + self.gamma * next_z.detach()

            # Compute loss
            loss = criterion(self.link(curr_z), self.link(z_t))

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            # Update state and index
            curr_index, curr_x, curr_y = next_index, next_x, next_y
            i += 1

            # Early stopping based on loss
            if loss.item() < self.epsilon:
                print(f'Ending optimization early at iteration {i}')
                break

        self.trained = True

class TDError(Exception):
    pass
