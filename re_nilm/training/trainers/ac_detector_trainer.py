"""AC detector trainer — fits Random Forest on Dataport AC sub-meter labels."""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import List, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from re_nilm.features.load import extract_ac_features
from re_nilm.training.base import AbstractTrainer

logger = logging.getLogger(__name__)

# Feature columns — must match ACDetector._DEFAULT_FEATURE_COLS and inference code
FEATURE_COLS: List[str] = [
    "corr_temp_all",
    "corr_temp_hot",
    "corr_rad_all",
    "season_balance",
    "thermal_balance",
    "summer_share",
    "daytime_share",
    "coeff_var",
    "acf_1h",
    "acf_24h",
    "afternoon_peak_ratio",
    "peak_summer_hour",
    "summer_vs_spring",
    "hot_load_ratio",
]

RF_PARAMS = dict(
    n_estimators=400,
    max_depth=8,
    min_samples_leaf=2,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1,
)

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent


def _build_feature_table_from_raw(df: pd.DataFrame) -> pd.DataFrame:
    """Build per-customer AC feature table from the raw training time series.

    Accepts the Dataport/all_sources format:
      columns: [type, source, dt_utc, glob_rad, value_kw_mean, id_customer, temp]
      type values: 'AC' (sub-meter), 'TOT' (aggregate load)

    Delegates label inference and feature extraction to the existing
    ac_actrainingfunctions.build_modeled_dataset(), then maps its output
    to the expected trainer format (adding a binary 'has_ac' column).
    """
    if str(_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(_REPO_ROOT))
    from model.ac_actrainingfunctions import (  # type: ignore[import]
        build_modeled_dataset,
        convert_f_to_c_if_needed,
    )

    needed = ["type", "source", "dt_utc", "glob_rad", "value_kw_mean", "id_customer", "temp"]
    df = df[needed].copy()
    df = df[df["type"].isin(["AC", "TOT"])].copy()
    df["dt_utc"] = pd.to_datetime(df["dt_utc"], utc=True, errors="coerce")
    df = df.dropna(subset=["dt_utc"])
    df = convert_f_to_c_if_needed(df)
    df = df.sort_values(["id_customer", "type", "dt_utc"]).reset_index(drop=True)

    modeled_df = build_modeled_dataset(df)
    if modeled_df.empty:
        raise ValueError("No usable customers found in raw training data after filtering")

    # target_name ∈ {"has_ac", "no_ac"} → binary has_ac column
    modeled_df["has_ac"] = (modeled_df["target_name"] == "has_ac").astype(int)
    return modeled_df


class ACDetectorTrainer(AbstractTrainer):
    """Train a Random Forest binary classifier for AC presence detection.

    Accepts either:
      - Raw time series training data (all_sources_load_with_weather.parquet format):
        columns [type, source, dt_utc, glob_rad, value_kw_mean, id_customer, temp].
        Feature extraction and label inference run automatically.
      - Pre-computed per-customer feature table with FEATURE_COLS + 'has_ac' column.

    Args:
        test_size: Fraction of data to hold out for evaluation.
        rf_params: sklearn RandomForestClassifier keyword arguments.
    """

    def __init__(
        self,
        test_size: float = 0.2,
        rf_params: Optional[dict] = None,
    ):
        self.test_size = test_size
        self.rf_params = rf_params or RF_PARAMS
        self.pipeline_: Optional[Pipeline] = None
        self._eval_results: dict = {}

    def fit(self, training_df: pd.DataFrame) -> None:
        """Fit on training data — raw time series or pre-computed feature table."""
        # Auto-detect raw time series format
        if "type" in training_df.columns and "value_kw_mean" in training_df.columns:
            logger.info("[ACDetectorTrainer] Raw time series detected — running feature extraction")
            training_df = _build_feature_table_from_raw(training_df)
            logger.info("[ACDetectorTrainer] Feature table: %d customers", len(training_df))

        missing = [c for c in FEATURE_COLS if c not in training_df.columns]
        if missing:
            raise ValueError(f"training_df missing feature columns: {missing}")
        if "has_ac" not in training_df.columns:
            raise ValueError("training_df must have a 'has_ac' column")

        X = training_df[FEATURE_COLS].to_numpy(dtype=float)
        y = training_df["has_ac"].astype(int).to_numpy()

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=42, stratify=y
        )

        pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("clf", RandomForestClassifier(**self.rf_params)),
        ])
        pipe.fit(X_train, y_train)

        y_pred = pipe.predict(X_test)
        self._eval_results = {
            "n_train": len(X_train),
            "n_test": len(X_test),
            "f1": float(f1_score(y_test, y_pred, average="weighted")),
            "report": classification_report(y_test, y_pred, target_names=["no_ac", "has_ac"]),
        }
        self.pipeline_ = pipe
        logger.info("[ACDetectorTrainer] Fitted. F1=%.3f on %d test samples", self._eval_results["f1"], len(X_test))

    def save(self, path: Path) -> None:
        if self.pipeline_ is None:
            raise RuntimeError("Call fit() before save()")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.pipeline_, path)
        logger.info("[ACDetectorTrainer] Model saved → %s", path)

    def evaluate(self) -> dict:
        return dict(self._eval_results)

    @classmethod
    def extract_features_from_timeseries(
        cls,
        customer_df: pd.DataFrame,
        weather_df: pd.DataFrame,
        inference_months: Optional[List[int]] = None,
        day_rad_threshold: float = 50.0,
        min_day_rows: int = 50,
    ) -> Optional[dict]:
        """Extract AC features from a single customer's time series + weather.

        This is the same feature extraction used at inference time (no train/serve skew).
        Returns None if insufficient data.
        """
        df = customer_df.copy()
        df["DT_UTC"] = pd.to_datetime(df["DT_UTC"], errors="coerce")
        df = df.dropna(subset=["DT_UTC"]).sort_values("DT_UTC")

        if inference_months:
            df = df[df["DT_UTC"].dt.month.isin(inference_months)]
        if df.empty:
            return None

        # Normalize to datetime64[us] — pandas 2.x merge_asof requires identical units
        df["DT_UTC"] = df["DT_UTC"].astype("datetime64[us]")
        wdf = weather_df.copy()
        wdf["dt_utc"] = wdf["dt_utc"].astype("datetime64[us]")
        wdf = wdf.sort_values("dt_utc")
        merged = pd.merge_asof(
            df.rename(columns={"DT_UTC": "_ts"}),
            wdf.rename(columns={"dt_utc": "_ts"}),
            on="_ts",
            direction="backward",
            tolerance=pd.Timedelta("1h"),
        )

        load = pd.to_numeric(merged["CONSO_KWH"], errors="coerce") * 4.0
        temp = pd.to_numeric(merged.get("t_2m_C", pd.Series(dtype=float)), errors="coerce")
        rad = pd.to_numeric(merged.get("global_rad_W", pd.Series(dtype=float)), errors="coerce")
        ts = merged["_ts"]

        return extract_ac_features(load, temp, rad, ts, day_rad_threshold=day_rad_threshold, min_day_rows=min_day_rows)
