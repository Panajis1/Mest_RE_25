"""Abstract base class for all appliance trainers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import pandas as pd


class AbstractTrainer(ABC):
    """Interface every trainer must implement.

    A trainer fits a model on labeled training data, evaluates it, and serializes
    the artifact to disk. Training is an offline one-time operation; inference
    never retrains.
    """

    @abstractmethod
    def fit(self, training_df: pd.DataFrame) -> None:
        """Fit the model on labeled training data.

        Args:
            training_df: Unified training table with features and labels.
        """

    @abstractmethod
    def save(self, path: Path) -> None:
        """Serialize the fitted model to path using joblib."""

    @abstractmethod
    def evaluate(self) -> dict:
        """Return evaluation metrics on the holdout set.

        Returns:
            Dict with at minimum {'accuracy', 'f1', 'n_train', 'n_test'}.
        """

    @classmethod
    def load(cls, path: Path) -> "AbstractTrainer":
        """Load a previously saved trainer artifact for evaluation only."""
        raise NotImplementedError(f"{cls.__name__} does not support load()")
