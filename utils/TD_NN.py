"""
TD Learning with a Two-Hidden-Layer Neural Network (Mini-Batch Training)

Date: 10/9/24
Author: Your Name
"""

import torch
from torch.utils.data import DataLoader, Dataset
from typing import Callable
from utils.NeuralNet import TwoHiddenLayerNN

class TransitionDataset(Dataset):
    """Custom Dataset for storing transitions."""
    def __init__(self, X: torch.Tensor, y: torch.Tensor, P: torch.Tensor):
        self.X = X
        self.y = y
        self.P = P
        self.n_samples = X.size(0)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        curr_x = self.X[idx]
        curr_y = self.y[idx]

        # Sample next state based on transition matrix P
        probs = self.P[idx]
        next_idx = torch.multinomial(probs, 1).item()
        next_x = self.X[next_idx]
        next_y = self.y[next_idx]

        return curr_x, curr_y, next_x, next_y

class TemporalDifferenceNN:
    def __init__(
            self,
            optimizer: str,
            input_size: int,
            output_size: int = 1,
            learning_rate: float = 0.01,
            gamma: float = 0,
            epsilon: float = 0,
            P: torch.Tensor = None,
            link: Callable[[torch.Tensor], torch.Tensor] = None,
            inv_link: Callable[[torch.Tensor], torch.Tensor] = None,
            betas: tuple[float, float] = (0.9, 0.999),
            batch_size: int = 32,
            random_state: int = None,
        ) -> None:
        # Set seed for reproducibility
        if random_state is not None:
            torch.manual_seed(random_state)

        # Initialize the neural network model
        self.nn = TwoHiddenLayerNN(
            input_size=input_size,
            output_size=output_size,
            optimizer=optimizer,
            learning_rate=learning_rate,
            betas=betas,
            batch_size=batch_size,
        )

        # TD Learning parameters
        self.gamma = gamma
        self.epsilon = epsilon
        self.P = P
        self.link = link if link else lambda x: x  # Identity function if None
        self.inv_link = inv_link if inv_link else lambda x: x  # Identity function if None
        self.batch_size = batch_size

        # Validate the transition matrix P
        if self.P is None:
            raise ValueError("Transition matrix P must be provided.")
        if not torch.allclose(self.P.sum(dim=1), torch.ones(self.P.size(0))):
            raise ValueError("Each row of the transition matrix P must sum to 1.")
        
    def fit(self, X: torch.Tensor, y: torch.Tensor, epochs: int = 100) -> None:
        n_samples = X.size(0)

        if y.size(0) != n_samples:
            raise ValueError("Ensure there are the same number of target samples as feature samples.")

        # Ensure inputs are tensors
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, dtype=torch.float32)
        if not isinstance(y, torch.Tensor):
            y = torch.tensor(y, dtype=torch.float32)

        # Create TransitionDataset and DataLoader for mini-batch training
        dataset = TransitionDataset(X, y, self.P)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.nn.model.train()

        for epoch in range(epochs):
            epoch_loss = 0.0
            for batch in dataloader:
                curr_x, curr_y, next_x, next_y = batch

                # Zero the gradients
                self.nn.optimizer.zero_grad()

                # Forward pass for current and next states
                curr_z = self.nn.model(curr_x)
                with torch.no_grad():
                    next_z = self.nn.model(next_x)

                # Compute reward
                r = self.inv_link(curr_y) - self.gamma * self.inv_link(next_y)

                # TD target
                z_t = r.unsqueeze(1) + self.gamma * next_z

                # Compute loss
                loss = self.nn.criterion(self.link(curr_z), self.link(z_t))

                # Backward pass and optimization
                loss.backward()
                self.nn.optimizer.step()

                epoch_loss += loss.item() * curr_x.size(0)

            # Early stopping based on loss
            av_epoch_loss = epoch_loss / len(dataloader)
            if av_epoch_loss < self.epsilon:
                print(f'Ending optimization early at epoch {epoch+1} with loss {av_epoch_loss}')
                break

        self.nn.trained = True
        print(f"Training finished. Final epoch loss: {av_epoch_loss}")

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        """Predict outputs for the given input X."""
        return self.nn.predict(X)

    def rmse(self, X: torch.Tensor, y: torch.Tensor) -> float:
        """Calculate the RMSE between the model predictions and targets."""
        return self.nn.rmse(X, y)

    def reset(self) -> None:
        """Reset NN weights."""
        self.nn.reset()
