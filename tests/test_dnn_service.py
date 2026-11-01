"""Offline tests for the PyTorch DNN regression service."""

import numpy as np
import pandas as pd
import pytest
import torch

from app.services.dnn_service import (
    DNNConfig,
    load_dnn_checkpoint,
    predict_dnn,
    save_dnn_checkpoint,
    train_dnn_regressor,
)


def _data():
    """Build a small deterministic regression dataset."""
    random = np.random.default_rng(42)
    features = pd.DataFrame(
        random.normal(size=(64, 3)), columns=["signal_a", "signal_b", "signal_c"]
    )
    target = 2 * features["signal_a"] - features["signal_c"]
    return features, target


def _config(**overrides):
    """Keep the tests small while exercising the model architecture and trainer."""
    values = {
        "hidden_units": (16, 8),
        "dropout_rates": (0.2, 0.1),
        "epochs": 30,
        "batch_size": 16,
        "patience": 5,
        "random_seed": 42,
    }
    values.update(overrides)
    return DNNConfig(**values)


def test_training_uses_validation_for_stopping_and_returns_losses():
    """Training returns finite histories and a model that predicts held-out rows."""
    features, target = _data()
    result = train_dnn_regressor(
        features.iloc[:40],
        target.iloc[:40],
        validation_data=(features.iloc[40:52], target.iloc[40:52]),
        config=_config(),
    )

    assert isinstance(result.model, torch.nn.Module)
    assert result.best_epoch <= len(result.train_losses)
    assert len(result.validation_losses) == len(result.train_losses)
    assert np.isfinite(result.train_losses).all()
    assert predict_dnn(result, features.iloc[52:]).shape == (12,)


def test_scaler_is_fitted_only_on_training_rows():
    """Validation outliers do not affect the training-fitted scaler mean."""
    features, target = _data()
    validation = features.iloc[40:48].copy() + 1000
    result = train_dnn_regressor(
        features.iloc[:40],
        target.iloc[:40],
        validation_data=(validation, target.iloc[40:48]),
        config=_config(epochs=3, patience=2),
    )

    np.testing.assert_allclose(
        result.scaler.mean_, features.iloc[:40].mean().values, rtol=1e-6, atol=1e-8
    )
    assert not np.allclose(result.scaler.mean_, validation.mean().values)


def test_model_checkpoint_round_trip_preserves_predictions(tmp_path):
    """Weights, architecture settings, and scaler reproduce inference results."""
    features, target = _data()
    result = train_dnn_regressor(
        features.iloc[:48], target.iloc[:48], config=_config(epochs=5)
    )
    expected = predict_dnn(result, features.iloc[48:])
    path = save_dnn_checkpoint(result, tmp_path / "models" / "dnn.pt")
    restored = load_dnn_checkpoint(path)

    np.testing.assert_allclose(predict_dnn(restored, features.iloc[48:]), expected)
    assert restored.config.hidden_units == (16, 8)


def test_validation_shape_mismatch_is_rejected():
    """Validation input must agree with the training feature width."""
    features, target = _data()

    with pytest.raises(ValueError, match="feature count"):
        train_dnn_regressor(
            features.iloc[:20],
            target.iloc[:20],
            validation_data=(features.iloc[20:30, :2], target.iloc[20:30]),
            config=_config(epochs=2),
        )
