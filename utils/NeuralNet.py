import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

class TwoHiddenLayerNN:
    def __init__(
            self, 
            input_size: int, 
            output_size: int = 1, 
            optimizer: str = 'sgd', 
            learning_rate: float = 0.01,
            betas: tuple[float, float] = (0.9, 0.999),
            batch_size: int = 32,
            epsilon: float = 0,
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
        
        self.batch_size = batch_size
        self.epsilon = epsilon
        self.trained = False

    def fit(self, X: torch.Tensor, y: torch.Tensor, epochs: int = 100) -> None:
        # Ensure inputs are tensors
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, dtype=torch.float32)
        if not isinstance(y, torch.Tensor):
            y = torch.tensor(y, dtype=torch.float32)

        # Create DataLoader for mini-batch training
        dataset = TensorDataset(X, y)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        for epoch in range(epochs):
            self.model.train()
            epoch_loss = 0.0
            for batch_X, batch_y in dataloader:
                # Zero the gradients
                self.optimizer.zero_grad()

                # Forward pass
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y.unsqueeze(1))

                # Backward pass and optimization
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item() * batch_X.size(0)
            
            av_epoch_loss = epoch_loss / len(dataloader)
            if av_epoch_loss < self.epsilon:
                print(f'Ending optimization early at epoch {epoch+1} with loss {av_epoch_loss}')
                break
        
        self.trained = True
        print(f"Training finished. Final epoch loss: {av_epoch_loss}")

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
        """Reset NN weights."""
        if not self.trained:
            return
        for layer in self.model.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
        self.trained = False

class NNError(Exception):
    pass
