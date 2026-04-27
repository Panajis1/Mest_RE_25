"""Abstract base class for all appliance detectors."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import pandas as pd


class AbstractDetector(ABC):
    """Interface every detector must implement.

    A detector answers: does this customer have appliance X?

    All detectors work on a single-customer basis via predict_customer().
    The streaming engine calls this per customer and collects results.
    """

    @abstractmethod
    def predict_customer(
        self,
        customer_df: pd.DataFrame,
        weather_df: pd.DataFrame,
        **context,
    ) -> dict | None:
        """Run detection for a single customer.

        Args:
            customer_df: 15-min time series [DT_UTC, CONSO_KWH, PROD_KWH, ID].
            weather_df: Aligned weather data [dt_utc, t_2m_C, global_rad_W].
            **context: Optional prior results (e.g. pv_result for battery detector).

        Returns:
            Dict with at minimum {customer_id, has_<X>, prob_<X>}, or None if
            the customer does not meet minimum data quality requirements.
        """

    @classmethod
    def load(cls, path: Path) -> "AbstractDetector":
        """Load a detector from a serialized model artifact (ML detectors only)."""
        raise NotImplementedError(f"{cls.__name__} does not support load()")

    def filter_detected(self, results: list[dict]) -> list[dict]:
        """Return only the results where the appliance was detected."""
        if not results:
            return []
        key = next((k for k in results[0] if k.startswith("has_")), None)
        if key is None:
            return []
        return [r for r in results if r.get(key, False)]
