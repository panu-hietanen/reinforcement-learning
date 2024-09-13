"""
Code containing a class for performing standardised Temporal Difference learning.

Uses methods from https://arxiv.org/abs/2404.15518v3

Date: 10/9/24
Author: Panu Hietanen
"""
import torch
from torch.optim import Adam, SGD
from typing import Callable
from abc import ABC, abstractmethod


class BaseTD(ABC):
    """Class to perform optimization using generalised TD methods with torch tensors."""

    def __init__(
            self,
            n_iter: int,
            P: torch.Tensor,
            link: Callable[[torch.Tensor], torch.Tensor],
            inv_link: Callable[[torch.Tensor], torch.Tensor],
            gamma: float,
            alpha: float,
            epsilon: float,
            random_state: int = None,
    ) -> None:
        self.n_iter = n_iter
        self.link = link
        self.inv_link = inv_link
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon

        # Set seed for reproducibility
        self.rng = torch.Generator()
        if random_state is not None:
            self.rng.manual_seed(random_state)

        # Ensure P matrix sums to one (along rows)
        if not torch.allclose(P.sum(dim=1), torch.ones(P.size(0))):
            raise ValueError("Each row of the transition matrix P must sum to 1.")
        self.P = P

        self.weights = None
        self.bias = None

    def sample_next_state(self, index: int) -> int:
        """Sample the next state based on the transition matrix P."""
        probs = self.P[index]
        next_state = torch.multinomial(probs, 1, generator=self.rng)
        return next_state.item()

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        if self.weights is None:
            raise TDError("Please use the fit function first!")
        return torch.matmul(X, self.weights) + self.bias

    def rmse(self, X: torch.Tensor, y: torch.Tensor) -> float:
        if self.weights is None:
            raise TDError("Please use the fit function first!")
        y_hat = self.predict(X)
        error = y_hat - y
        return torch.sqrt(torch.mean(torch.pow(error, 2)))

    def reset(self) -> None:
        """Reset the model's weights and bias."""
        self.weights = self.bias = None

    @abstractmethod
    def fit(self, X: torch.Tensor, y: torch.Tensor) -> None:
        pass


class TD_SGD(BaseTD):
    def fit(self, X: torch.Tensor, y: torch.Tensor) -> None:
        n_samples, n_features = X.shape

        if y.size(0) != n_samples:
            raise TDError("Ensure there are the same number of target samples as feature samples.")

        X_bias = torch.cat([torch.ones(n_samples, 1), X], dim=1)
        w = torch.zeros(n_features + 1)

        curr_index = torch.randint(0, n_samples, (1,), generator=self.rng).item()
        curr_x = X_bias[curr_index]
        curr_y = y[curr_index]

        i = 0
        grad = torch.full_like(w, float(0))

        while i < self.n_iter:
            # Next state samples
            next_index = self.sample_next_state(curr_index)
            next_x = X_bias[next_index]
            next_y = y[next_index]

            # Find predictions
            curr_z = torch.dot(curr_x, w)
            next_z = torch.dot(next_x, w)

            # Find rewards
            r = self.inv_link(curr_y) - self.gamma * self.inv_link(next_y)

            # TD target
            z_t = r + self.gamma * next_z

            # Find gradient
            grad = (self.link(curr_z) - self.link(z_t)) * curr_x

            # Update weights
            w -= self.alpha * grad

            # Update state and index
            curr_index, curr_x, curr_y = next_index, next_x, next_y
            i += 1

            # Early stopping if the gradient is small enough
            if torch.norm(self.alpha * grad) < self.epsilon:
                print(f'Ending optimization early at iteration {i}')
                break

        self.weights = w[1:]  # Separate weights from bias
        self.bias = w[0]


class TD_Adam(BaseTD):
    def __init__(
                self,
                n_iter: int,
                P: torch.Tensor,
                link: Callable[[torch.Tensor], torch.Tensor],
                inv_link: Callable[[torch.Tensor], torch.Tensor],
                gamma: float,
                alpha: float,
                epsilon: float,
                betas: tuple[float, float] = (0.9, 0.999),
                random_state: int = None,
        ) -> None:
        super(TD_Adam, self).__init__(n_iter, P, link, inv_link, gamma, alpha, epsilon, random_state)
        self.betas = betas


    def fit(self, X: torch.Tensor, y: torch.Tensor) -> None:
        n_samples, n_features = X.shape

        if y.size(0) != n_samples:
            raise TDError("Ensure there are the same number of target samples as feature samples.")

        X_bias = torch.cat([torch.ones(n_samples, 1), X], dim=1)
        w = torch.zeros(n_features + 1, requires_grad=True)

        # Initialise optimizer
        optimizer = Adam([w], lr=self.alpha, betas=self.betas)

        curr_index = torch.randint(0, n_samples, (1,), generator=self.rng).item()
        curr_x = X_bias[curr_index]
        curr_y = y[curr_index]

        i = 0

        while i < self.n_iter:
            # Next state samples
            next_index = self.sample_next_state(curr_index)
            next_x = X_bias[next_index]
            next_y = y[next_index]

            optimizer.zero_grad()

            # Find predictions
            curr_z = torch.dot(curr_x, w)
            next_z = torch.dot(next_x, w)

            # Find rewards
            r = self.inv_link(curr_y) - self.gamma * self.inv_link(next_y)

            # TD target
            z_t = r + self.gamma * next_z

            # Find loss
            loss = (self.link(curr_z) - self.link(z_t)) ** 2 / 2

            # Update weights
            loss.backward()
            optimizer.step()

            # Update state and index
            curr_index, curr_x, curr_y = next_index, next_x, next_y
            i += 1

            # Early stopping based on loss
            if loss.item() < self.epsilon:
                print(f'Ending optimization early at iteration {i}')
                break

        self.weights = w.detach()[1:]  # Separate weights from bias
        self.bias = w.detach()[0]


class TDError(Exception):
    pass
