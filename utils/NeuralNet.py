"""
Code containing a class for training a Neural Network for regression with two fully connected hidden layers.

Date: 10/9/24
Author: Panu Hietanen
"""
import torch
from torch import optim, nn
from torch.utils.data import DataLoader, TensorDataset
from abc import ABC, abstractmethod

class BaseThreeLayerFCNN(nn.Module):
    def __init__(
            self, 
            batch_size: int,
            lr: float,
            n_epochs: int,
            input_size: int, 
            hidden_size: int = 64, 
            output_size: int = 1
            ):
        super(BaseThreeLayerFCNN, self).__init__()

        self.device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )

        # First fully connected layer
        self.fc1 = nn.Linear(input_size, hidden_size)
        # Activation function (ReLU)
        self.relu = nn.ReLU()
        # Second fully connected layer
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        # Third fully connected layer (output layer)
        self.fc3 = nn.Linear(hidden_size, output_size)

        self.batch_size = batch_size
        self.lr = lr
        self.n_epochs = n_epochs
        self.input_size = input_size
        self.hidden_size1 = hidden_size
        self.hidden_size2 = hidden_size
        self.output_size = output_size

        self.trained = False

    # Forward method
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        return out

    def evaluate(self, X: torch.Tensor, y: torch.Tensor) -> float:
        """Evaluate the performance of the trained neural net."""
        if not self.trained:
            print("Weights have not yet been optimized.")
            return None
        test_data = TensorDataset(X, y)
        test_loader = DataLoader(test_data, batch_size=self.batch_size, shuffle=False)

        self.eval()  # Set the model to evaluation mode
        total_loss = 0.0
        criterion = nn.MSELoss()  # Use MSE for evaluation

        with torch.no_grad():  # No gradient calculation during evaluation
            for data, target in test_loader:
                output = self(data)
                loss = criterion(output, target.unsqueeze(1))
                total_loss += loss.item()

        avg_test_loss = total_loss / len(test_loader)
        return avg_test_loss
    
    def rmse(self, X: torch.Tensor, y: torch.Tensor) -> float:
        if not self.trained:
            raise NNError("Please use the fit function first!")
        y_hat = self.predict(X)
        error = y_hat - y
        return torch.sqrt(torch.mean(torch.pow(error, 2)))

    def reset(self) -> None:
        if self.trained:
            for layer in self.children():
                if hasattr(layer, 'reset_parameters'):
                    layer.reset_parameters()
        else:
            print("Weights have not yet been optimized.")

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        self.eval()

        with torch.no_grad():
            output = self(X)

        return output

class ThreeLayerFCNN_Adam(BaseThreeLayerFCNN):
    def __init__(
            self, 
            batch_size: int,
            lr: float,
            n_epochs: int,
            input_size: int, 
            hidden_size: int = 64, 
            output_size: int = 1,
            betas: tuple = (0.9, 0.999),
        ) -> None:
        super(ThreeLayerFCNN_Adam, self).__init__(batch_size, lr, n_epochs, input_size, hidden_size, output_size)
        self.betas = betas

    def fit(self, X: torch.Tensor, y: torch.Tensor) -> None:
        """Fit training data using Adam optimizer."""
        train_data = TensorDataset(X, y)
        train_loader = DataLoader(train_data, batch_size=self.batch_size, shuffle=True)

        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.parameters(), lr=self.lr, betas=self.betas)

        for epoch in range(self.n_epochs):
            self.train()  # Set the model to training mode
            epoch_loss = 0.0  # Variable to accumulate the epoch's loss

            for batch_idx, (data, target) in enumerate(train_loader):
                optimizer.zero_grad()  # Zero the gradients

                # Forward pass
                output = self(data)

                # Calculate the loss
                loss = criterion(output, target.unsqueeze(1))  # Match dimensions of output and target

                # Backward pass and optimization
                loss.backward()
                optimizer.step()

                # Accumulate loss for this epoch
                epoch_loss += loss.item()

            # Average loss for the epoch
            avg_loss = epoch_loss / len(train_loader)
            print(f'Epoch [{epoch+1}/{self.n_epochs}], Loss: {avg_loss:.4f}')
        self.trained = True

class ThreeLayerFCNN_SGD(BaseThreeLayerFCNN):
    def fit(self, X: torch.Tensor, y: torch.Tensor) -> None:
        train_data = TensorDataset(X, y)
        train_loader = DataLoader(train_data, batch_size=self.batch_size, shuffle=True)

        criterion = nn.MSELoss()
        optimizer = optim.SGD(self.parameters(), lr=self.lr)

        for epoch in range(self.n_epochs):
            self.train()  # Set the model to training mode
            epoch_loss = 0.0  # Variable to accumulate the epoch's loss

            for batch_idx, (data, target) in enumerate(train_loader):
                optimizer.zero_grad()  # Zero the gradients

                # Forward pass
                output = self(data)

                # Calculate the loss
                loss = criterion(output, target.unsqueeze(1))  # Match dimensions of output and target

                # Backward pass and optimization
                loss.backward()
                optimizer.step()

                # Accumulate loss for this epoch
                epoch_loss += loss.item()

            # Average loss for the epoch
            avg_loss = epoch_loss / len(train_loader)
            print(f'Epoch [{epoch+1}/{self.n_epochs}], Loss: {avg_loss:.4f}')
        self.trained = True


class NNError(Exception):
    pass