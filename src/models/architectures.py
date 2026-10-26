"""Deep Learning Tabular Architectures and PyTorch Training Harness.

Implements:
- Standard TabularMLP (dense feed-forward with Dropout and BatchNorm).
- ResidualTabularMLP (ResNet-style architecture with skip-connections and LayerNorm).
- Early stopping PyTorch training harness with seed reproducibility and validation loss tracking.
- Safe graceful fallback if PyTorch is missing or unsupported on the platform.
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

from src.utils.logger import get_logger
from src.utils.seed import set_seed

logger = get_logger("models.architectures")

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    _TORCH_AVAILABLE = True
except ImportError:
    torch = None
    nn = None
    optim = None
    _TORCH_AVAILABLE = False


if _TORCH_AVAILABLE and nn is not None:

    class TabularMLP(nn.Module):
        """Standard Feed-Forward Tabular Neural Network with Regularization."""

        def __init__(
            self,
            input_dim: int,
            hidden_dims: Optional[List[int]] = None,
            output_dim: int = 1,
            dropout_rate: float = 0.20,
        ) -> None:
            super().__init__()
            if hidden_dims is None:
                hidden_dims = [64, 32]

            layers: List[nn.Module] = []
            prev_dim = input_dim

            for h_dim in hidden_dims:
                layers.append(nn.Linear(prev_dim, h_dim))
                layers.append(nn.BatchNorm1d(h_dim))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout_rate))
                prev_dim = h_dim

            layers.append(nn.Linear(prev_dim, output_dim))
            self.network = nn.Sequential(*layers)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.network(x)


    class ResidualBlock(nn.Module):
        """Residual skip-connection block for tabular representation learning."""

        def __init__(self, dim: int, dropout_rate: float = 0.20) -> None:
            super().__init__()
            self.block = nn.Sequential(
                nn.Linear(dim, dim),
                nn.LayerNorm(dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(dim, dim),
                nn.LayerNorm(dim),
            )
            self.relu = nn.ReLU()

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.relu(x + self.block(x))


    class ResidualTabularMLP(nn.Module):
        """ResNet-style Tabular Architecture with Skip Connections."""

        def __init__(
            self,
            input_dim: int,
            hidden_dim: int = 64,
            num_blocks: int = 2,
            output_dim: int = 1,
            dropout_rate: float = 0.20,
        ) -> None:
            super().__init__()
            self.input_layer = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
            )
            self.blocks = nn.ModuleList([
                ResidualBlock(hidden_dim, dropout_rate=dropout_rate) for _ in range(num_blocks)
            ])
            self.head = nn.Linear(hidden_dim, output_dim)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            h = self.input_layer(x)
            for block in self.blocks:
                h = block(h)
            return self.head(h)


    class PyTorchTabularEstimator:
        """Sklearn-compatible wrapper for PyTorch tabular regression and classification."""

        def __init__(
            self,
            arch: str = "mlp",  # "mlp" or "res_mlp"
            track: str = "regression",  # "regression" or "classification"
            hidden_dims: Optional[List[int]] = None,
            dropout_rate: float = 0.20,
            learning_rate: float = 1e-3,
            weight_decay: float = 1e-4,
            batch_size: int = 64,
            max_epochs: int = 40,
            patience: int = 6,
            seed: int = 42,
        ) -> None:
            self.arch = arch
            self.track = track
            self.hidden_dims = hidden_dims or [64, 32]
            self.dropout_rate = dropout_rate
            self.learning_rate = learning_rate
            self.weight_decay = weight_decay
            self.batch_size = batch_size
            self.max_epochs = max_epochs
            self.patience = patience
            self.seed = seed
            self.model: Optional[nn.Module] = None
            self.classes_: Optional[np.ndarray] = None
            self.history_: Dict[str, List[float]] = {"train_loss": [], "val_loss": []}
            self.is_fitted_ = False

        def __sklearn_is_fitted__(self) -> bool:
            return self.is_fitted_

        def fit(
            self,
            X: Union[pd.DataFrame, np.ndarray],
            y: Union[pd.Series, np.ndarray],
            X_val: Optional[Union[pd.DataFrame, np.ndarray]] = None,
            y_val: Optional[Union[pd.Series, np.ndarray]] = None,
        ) -> "PyTorchTabularEstimator":
            set_seed(self.seed)
            X_arr = X.values if isinstance(X, pd.DataFrame) else np.asarray(X, dtype=np.float32)
            y_arr = y.values if isinstance(y, pd.Series) else np.asarray(y)

            input_dim = X_arr.shape[1]

            if self.track == "classification":
                self.classes_, y_mapped = np.unique(y_arr, return_inverse=True)
                output_dim = len(self.classes_)
                y_tensor = torch.tensor(y_mapped, dtype=torch.long)
                criterion = nn.CrossEntropyLoss()
            else:
                output_dim = 1
                y_tensor = torch.tensor(y_arr, dtype=torch.float32).unsqueeze(1)
                criterion = nn.MSELoss()

            X_tensor = torch.tensor(X_arr, dtype=torch.float32)
            dataset = TensorDataset(X_tensor, y_tensor)
            loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

            # Build model
            if self.arch == "res_mlp":
                self.model = ResidualTabularMLP(
                    input_dim=input_dim,
                    hidden_dim=self.hidden_dims[0] if self.hidden_dims else 64,
                    num_blocks=len(self.hidden_dims) if self.hidden_dims else 2,
                    output_dim=output_dim,
                    dropout_rate=self.dropout_rate,
                )
            else:
                self.model = TabularMLP(
                    input_dim=input_dim,
                    hidden_dims=self.hidden_dims,
                    output_dim=output_dim,
                    dropout_rate=self.dropout_rate,
                )

            optimizer = optim.AdamW(
                self.model.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
            )

            # Validation data handling
            has_val = X_val is not None and y_val is not None
            if has_val:
                X_val_arr = X_val.values if isinstance(X_val, pd.DataFrame) else np.asarray(X_val, dtype=np.float32)
                y_val_arr = y_val.values if isinstance(y_val, pd.Series) else np.asarray(y_val)
                if self.track == "classification":
                    y_val_mapped = np.array([np.where(self.classes_ == c)[0][0] for c in y_val_arr])
                    y_val_tensor = torch.tensor(y_val_mapped, dtype=torch.long)
                else:
                    y_val_tensor = torch.tensor(y_val_arr, dtype=torch.float32).unsqueeze(1)
                X_val_tensor = torch.tensor(X_val_arr, dtype=torch.float32)

            best_val_loss = float("inf")
            best_state = None
            epochs_no_improve = 0

            self.model.train()
            for epoch in range(self.max_epochs):
                epoch_loss = 0.0
                for batch_x, batch_y in loader:
                    optimizer.zero_grad()
                    out = self.model(batch_x)
                    loss = criterion(out, batch_y)
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item() * len(batch_x)

                epoch_loss /= len(dataset)
                self.history_["train_loss"].append(epoch_loss)

                # Validation & early stopping
                if has_val:
                    self.model.eval()
                    with torch.no_grad():
                        val_out = self.model(X_val_tensor)
                        val_loss = criterion(val_out, y_val_tensor).item()
                    self.history_["val_loss"].append(val_loss)
                    self.model.train()

                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                        epochs_no_improve = 0
                    else:
                        epochs_no_improve += 1
                        if epochs_no_improve >= self.patience:
                            logger.debug("Early stopping triggered at epoch %d", epoch + 1)
                            break

            if best_state is not None:
                self.model.load_state_dict(best_state)

            self.model.eval()
            self.is_fitted_ = True
            return self

        def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
            if not self.is_fitted_ or self.model is None:
                raise RuntimeError("PyTorch estimator is not fitted.")
            X_arr = X.values if isinstance(X, pd.DataFrame) else np.asarray(X, dtype=np.float32)
            X_tensor = torch.tensor(X_arr, dtype=torch.float32)

            self.model.eval()
            with torch.no_grad():
                out = self.model(X_tensor)
                if self.track == "classification":
                    preds = torch.argmax(out, dim=1).cpu().numpy()
                    return self.classes_[preds] if self.classes_ is not None else preds
                else:
                    return out.squeeze(1).cpu().numpy()

        def predict_proba(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
            if self.track != "classification" or not self.is_fitted_ or self.model is None:
                raise RuntimeError("predict_proba is only available for fitted classification estimators.")
            X_arr = X.values if isinstance(X, pd.DataFrame) else np.asarray(X, dtype=np.float32)
            X_tensor = torch.tensor(X_arr, dtype=torch.float32)

            self.model.eval()
            with torch.no_grad():
                out = self.model(X_tensor)
                probs = torch.softmax(out, dim=1).cpu().numpy()
                return probs

else:
    # Graceful fallback placeholder
    class TabularMLP:  # type: ignore
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise ImportError("PyTorch is required for TabularMLP.")

    class ResidualTabularMLP:  # type: ignore
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise ImportError("PyTorch is required for ResidualTabularMLP.")

    class PyTorchTabularEstimator:  # type: ignore
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise ImportError("PyTorch is required for PyTorchTabularEstimator.")
