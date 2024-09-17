import torch
import torch.nn as nn
import torch.optim as optim

class TwoHiddenLayerNN:
    def __init__(
            self, 
            input_size: int, 
            output_size: int = 1, 
            optimizer: str = 'sgd', 
            learning_rate: float = 0.01,
            betas: tuple[float, float] = (0.9, 0.999),
        ) -> None:
        # Define the neural network architecture
        self.model = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, output_size)
        )

        # Mean Squared Error loss for regression
        self.criterion = nn.MSELoss()

        # Select optimizer
        if optimizer == 'sgd':
            self.optimizer = optim.SGD(self.model.parameters(), lr=learning_rate)
        elif optimizer == 'adam':
            self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, betas=betas)
        else:
            raise ValueError("Optimizer must be 'sgd' or 'adam'")
        
        self.trained = False

    def fit(self, X: torch.Tensor, y: torch.Tensor, epochs: int = 100) -> None:
        # Ensure inputs are tensors
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, dtype=torch.float32)
        if not isinstance(y, torch.Tensor):
            y = torch.tensor(y, dtype=torch.float32)

        for epoch in range(epochs):
            # Zero the gradients
            self.optimizer.zero_grad()

            # Forward pass
            outputs = self.model(X)
            loss = self.criterion(outputs, y.unsqueeze(1))

            # Backward pass and optimization
            loss.backward()
            self.optimizer.step()
        
        self.trained = True

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        # Ensure input is a tensor
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, dtype=torch.float32)
        if not self.trained:
            raise NNError('Model has not been trained.')
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X)
        return outputs

    def rmse(self, X: torch.Tensor, y: torch.Tensor) -> float:
        # Ensure inputs are tensors
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, dtype=torch.float32)
        if not isinstance(y, torch.Tensor):
            y = torch.tensor(y, dtype=torch.float32)
        if not self.trained:
            raise NNError('Model has not been trained.')
        with torch.no_grad():
            outputs = self.predict(X)
            mse_loss = self.criterion(outputs, y.unsqueeze(1))
            rmse = torch.sqrt(mse_loss)
        return rmse.item()
    
    def reset(self) -> None:
        if not self.trained:
            raise NNError('Model has not been trained.')
        for layer in self.model.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()

class NNError(Exception):
    pass
