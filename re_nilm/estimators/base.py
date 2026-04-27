"""Abstract base class for all appliance estimators (capacity, disaggregation)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import pandas as pd


class AbstractEstimator(ABC):
    """Interface every estimator must implement.

    An estimator quantifies appliance characteristics for a customer that was
    already flagged as positive by a detector.
    """

    @abstractmethod
    def estimate(
        self,
        customer_ts: pd.DataFrame,
        detection_result: dict,
        weather: pd.DataFrame,
    ) -> dict | None:
        """Estimate appliance capacity / disaggregated load for a single customer.

        Args:
            customer_ts: 15-min time series [DT_UTC, CONSO_KWH, PROD_KWH, ID].
            detection_result: Dict returned by the corresponding detector.
            weather: Aligned weather data [dt_utc, t_2m_C, global_rad_W].

        Returns:
            Dict of capacity/quantity estimates, or None on failure.
        """

    @classmethod
    def load(cls, path: Path) -> "AbstractEstimator":
        """Load a serialized estimator model from path (ML estimators only)."""
        raise NotImplementedError(f"{cls.__name__} does not support load()")
