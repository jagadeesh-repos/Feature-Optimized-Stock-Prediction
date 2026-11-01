"""Train the notebook-shaped regression DNN using PyTorch, not TensorFlow.

The hidden-layer widths, dropout rates, optimizer, loss, batch size, epoch limit,
and patience follow the user-supplied code. Standardization is fit only on the
training fold. Validation data controls early stopping; final test data must not
be passed here. These choices are not claimed as paper-verified settings.
"""

from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


@dataclass(frozen=True)
class DNNConfig:
    """Architecture and training parameters based on the supplied notebook."""

    hidden_units: tuple[int, int] = (128, 64)
    dropout_rates: tuple[float, float] = (0.2, 0.1)
    learning_rate: float = 1e-3
    epochs: int = 200
    batch_size: int = 64
    patience: int = 10
    random_seed: int = 42

    def __post_init__(self) -> None:
        """Reject invalid dimensions and optimization settings."""
        if len(self.hidden_units) != 2 or any(width < 1 for width in self.hidden_units):
            raise ValueError("hidden_units must contain two positive layer widths")
        if len(self.dropout_rates) != 2 or any(
            not 0 <= rate < 1 for rate in self.dropout_rates
        ):
            raise ValueError("dropout_rates must contain two values in [0, 1)")
        if self.learning_rate <= 0 or self.epochs < 1 or self.batch_size < 1:
            raise ValueError("learning_rate, epochs, and batch_size must be positive")
        if self.patience < 1:
            raise ValueError("patience must be positive")


class StockDNN(nn.Module):
    """Fully connected regression network with ReLU and dropout layers."""

    def __init__(self, input_features: int, config: DNNConfig):
        super().__init__()
        if input_features < 1:
            raise ValueError("input_features must be positive")
        first_width, second_width = config.hidden_units
        first_dropout, second_dropout = config.dropout_rates
        self.network = nn.Sequential(
            nn.Linear(input_features, first_width),
            nn.ReLU(),
            nn.Dropout(first_dropout),
            nn.Linear(first_width, second_width),
            nn.ReLU(),
            nn.Dropout(second_dropout),
            nn.Linear(second_width, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Return one continuous regression estimate per row."""
        return self.network(features).squeeze(-1)


@dataclass
class DNNTrainingResult:
    """Fitted DNN, training-only scaler, and epoch-level loss history."""

    model: StockDNN
    scaler: StandardScaler
    train_losses: list[float]
    validation_losses: list[float]
    best_epoch: int
    config: DNNConfig


def _feature_matrix(values: Any, name: str) -> np.ndarray:
    """Convert a tabular input to a finite float32 matrix."""
    try:
        matrix = np.asarray(values, dtype=np.float32)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must contain numeric features") from error
    if matrix.ndim != 2 or matrix.shape[0] < 1 or matrix.shape[1] < 1:
        raise ValueError(f"{name} must have at least one row and one feature")
    if not np.isfinite(matrix).all():
        raise ValueError(f"{name} must not contain missing or infinite values")
    return matrix


def _regression_target(values: Any, rows: int, name: str) -> np.ndarray:
    """Validate one continuous target per input row."""
    target = np.asarray(values, dtype=np.float32)
    if target.ndim != 1 or target.shape[0] != rows:
        raise ValueError(f"{name} must be one-dimensional and match feature rows")
    if not np.isfinite(target).all():
        raise ValueError(f"{name} must not contain missing or infinite values")
    return target


def train_dnn_regressor(
    X_train: pd.DataFrame | np.ndarray,
    y_train: Any,
    *,
    validation_data: tuple[pd.DataFrame | np.ndarray, Any] | None = None,
    config: DNNConfig = DNNConfig(),
) -> DNNTrainingResult:
    """Fit a PyTorch DNN, scaling on training rows and early stopping on validation."""
    train_features = _feature_matrix(X_train, "X_train")
    train_target = _regression_target(y_train, train_features.shape[0], "y_train")
    if isinstance(X_train, pd.DataFrame) and not X_train.columns.is_unique:
        raise ValueError("X_train column names must be unique")

    validation_features = None
    validation_target = None
    if validation_data is not None:
        validation_features = _feature_matrix(validation_data[0], "X_validation")
        validation_target = _regression_target(
            validation_data[1], validation_features.shape[0], "y_validation"
        )
        if validation_features.shape[1] != train_features.shape[1]:
            raise ValueError(
                "validation feature count must match training feature count"
            )
        if (
            isinstance(X_train, pd.DataFrame)
            and isinstance(validation_data[0], pd.DataFrame)
            and list(validation_data[0].columns) != list(X_train.columns)
        ):
            raise ValueError("validation columns must match training columns in order")

    torch.manual_seed(config.random_seed)
    scaler = StandardScaler()
    scaled_train = scaler.fit_transform(train_features).astype(np.float32)
    scaled_validation = (
        scaler.transform(validation_features).astype(np.float32)
        if validation_features is not None
        else None
    )

    train_dataset = TensorDataset(
        torch.from_numpy(scaled_train), torch.from_numpy(train_target)
    )
    generator = torch.Generator().manual_seed(config.random_seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,
    )

    model = StockDNN(train_features.shape[1], config)
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    validation_tensor = (
        torch.from_numpy(scaled_validation) if scaled_validation is not None else None
    )
    validation_target_tensor = (
        torch.from_numpy(validation_target) if validation_target is not None else None
    )

    train_losses: list[float] = []
    validation_losses: list[float] = []
    best_state = deepcopy(model.state_dict())
    best_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0

    for epoch in range(config.epochs):
        model.train()
        total_loss = 0.0
        observations = 0
        for batch_features, batch_target in train_loader:
            optimizer.zero_grad(set_to_none=True)
            predictions = model(batch_features)
            loss = loss_function(predictions, batch_target)
            loss.backward()
            optimizer.step()
            batch_size = batch_features.shape[0]
            total_loss += loss.item() * batch_size
            observations += batch_size
        train_losses.append(total_loss / observations)

        if validation_tensor is None or validation_target_tensor is None:
            best_state = deepcopy(model.state_dict())
            best_epoch = epoch + 1
            continue

        model.eval()
        with torch.no_grad():
            validation_loss = loss_function(
                model(validation_tensor), validation_target_tensor
            ).item()
        validation_losses.append(validation_loss)

        if validation_loss < best_loss:
            best_loss = validation_loss
            best_state = deepcopy(model.state_dict())
            best_epoch = epoch + 1
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= config.patience:
                break

    model.load_state_dict(best_state)
    model.eval()
    return DNNTrainingResult(
        model=model,
        scaler=scaler,
        train_losses=train_losses,
        validation_losses=validation_losses,
        best_epoch=best_epoch,
        config=config,
    )


def predict_dnn(
    result: DNNTrainingResult, features: pd.DataFrame | np.ndarray
) -> np.ndarray:
    """Predict continuous targets using the scaler fitted on training data."""
    matrix = _feature_matrix(features, "features")
    if matrix.shape[1] != result.scaler.n_features_in_:
        raise ValueError("prediction feature count must match training feature count")
    scaled = result.scaler.transform(matrix).astype(np.float32)
    result.model.eval()
    with torch.no_grad():
        return result.model(torch.from_numpy(scaled)).cpu().numpy()


def save_dnn_checkpoint(result: DNNTrainingResult, path: str | Path) -> Path:
    """Save state dict, architecture settings, and train-only scaler statistics."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "state_dict": result.model.state_dict(),
        "input_features": int(result.scaler.n_features_in_),
        "config": asdict(result.config),
        "scaler_mean": result.scaler.mean_.tolist(),
        "scaler_scale": result.scaler.scale_.tolist(),
        "best_epoch": result.best_epoch,
        "train_losses": result.train_losses,
        "validation_losses": result.validation_losses,
    }
    torch.save(checkpoint, destination)
    return destination


def load_dnn_checkpoint(path: str | Path) -> DNNTrainingResult:
    """Rebuild a DNN and scaler from a local state-dict checkpoint."""
    checkpoint = torch.load(Path(path), map_location="cpu", weights_only=True)
    config_values = checkpoint["config"]
    config_values["hidden_units"] = tuple(config_values["hidden_units"])
    config_values["dropout_rates"] = tuple(config_values["dropout_rates"])
    config = DNNConfig(**config_values)
    model = StockDNN(checkpoint["input_features"], config)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    scaler = StandardScaler()
    scaler.mean_ = np.asarray(checkpoint["scaler_mean"], dtype=np.float64)
    scaler.scale_ = np.asarray(checkpoint["scaler_scale"], dtype=np.float64)
    scaler.var_ = scaler.scale_**2
    scaler.n_features_in_ = checkpoint["input_features"]
    scaler.n_samples_seen_ = 1
    return DNNTrainingResult(
        model=model,
        scaler=scaler,
        train_losses=list(checkpoint["train_losses"]),
        validation_losses=list(checkpoint["validation_losses"]),
        best_epoch=int(checkpoint["best_epoch"]),
        config=config,
    )
