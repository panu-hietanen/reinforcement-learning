"""
Code containing a class for performing standardised Temporal Difference learning.

Uses methods from https://arxiv.org/abs/2404.15518v3

Date: 10/9/24
Author: Panu Hietanen
"""

import numpy as np
from typing import Callable

class TD:
    """Class to perform optimisation using generalised TD methods."""

    def __init__(
            self,
            n_iter: int, 
            P: np.ndarray,
            link: Callable[[np.ndarray], np.ndarray], 
            inv_link: Callable[[np.ndarray], np.ndarray], 
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

        # Set seed
        self.rng = np.random.default_rng(random_state)

        # Ensure P matrix sums to one
        if not np.allclose(P.sum(axis=1), 1):
            raise ValueError("Each row of the transition matrix P must sum to 1.")
        self.P = P

        self.weights = None
        self.bias = None

    def sample_next_state(self, index: int) -> int:
        probs = self.P[index]
        next = self.rng.choice(range(self.P.shape[1]), p=probs)
        return int(next)


    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        n_samples, n_features = X.shape

        X_bias = np.c_[np.ones(n_samples), X]

        w = np.zeros(n_features + 1)
    
        curr_index = self.rng.integers(low=0, high=n_samples)
        curr_x = X_bias[curr_index]
        curr_y = y[curr_index]
    
        i = 0
        grad = np.ones_like(w) * np.inf
    
        while i < self.n_iter and np.linalg.norm(self.alpha * grad, 2) > self.epsilon:
            # Next state samples
            next_index = self.sample_next_state(index=curr_index)
            next_x = X_bias[next_index]
            next_y = y[next_index]
    
            # Find predictions
            curr_z = np.dot(curr_x, w)
            next_z = np.dot(next_x, w)
    
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
    
        self.weights = w[1:]
        self.bias = w[0]


    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.weights is None:
            raise TDError("Please use the fit function first!")
        return np.dot(X, self.weights) + self.bias
    

    def rmse(self, X: np.ndarray, y: np.ndarray) -> float:
        if self.weights is None:
            raise TDError("Please use the fit function first!")
        y_hat = self.predict(X)

        error = y_hat - y

        return np.sqrt(np.mean(np.power(error, 2)))

        
class TDError(Exception):
    pass