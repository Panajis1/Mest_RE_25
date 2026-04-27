"""Abstract base for feature extractors."""

from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


class AbstractFeatureExtractor(ABC):
    """Base class for all feature extractors.

    Subclasses implement extract() which takes a per-customer DataFrame and
    returns a dict of feature_name → float (or None if extraction fails).
    """

    @abstractmethod
    def extract(self, customer_df: pd.DataFrame, weather_df: pd.DataFrame) -> dict | None:
        """Extract features for a single customer.

        Args:
            customer_df: 15-min load data with at minimum [DT_UTC, CONSO_KWH, PROD_KWH].
            weather_df: Weather data with [dt_utc, t_2m_C, global_rad_W].

        Returns:
            Dict of feature_name → float, or None if the customer does not meet
            minimum quality requirements.
        """
