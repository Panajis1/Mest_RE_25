"""HP detector trainer — fits Random Forest 3-class on Dataport HP sub-meter labels."""

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

from re_nilm.features.load import extract_hp_features
from re_nilm.training.base import AbstractTrainer

logger = logging.getLogger(__name__)

FEATURE_COLS: List[str] = [
    "corr_temp_all",
    "corr_temp_cold",
    "corr_temp_hot",
    "season_balance",
    "thermal_balance",
    "coeff_var",
    "acf_1h",
    "acf_24h",
]

# Class labels matching HeatPumpDetector._INT_TO_LABEL
LABEL_TO_INT = {"no_hp": 0, "winter_hp": 1, "summer_hp": 2}
INT_TO_LABEL = {v: k for k, v in LABEL_TO_INT.items()}

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
    """Build per-customer HP feature table from the raw training time series.

    Accepts the Dataport/all_sources format:
      columns: [type, source, dt_utc, glob_rad, value_kw_mean, id_customer, temp]
      type values: 'HP' (sub-meter), 'TOT' (aggregate load)

    Delegates label inference (winter_hp / summer_hp / no_hp) and feature
    extraction to hp_detection_functions.build_modeled_dataset(), then maps
    its output to the expected trainer format (adding hp_type / hp_label columns).
    """
    if str(_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(_REPO_ROOT))
    from hp_model.hp_detection_functions import (  # type: ignore[import]
        build_modeled_dataset,
        convert_f_to_c_if_needed,
    )

    needed = ["type", "source", "dt_utc", "glob_rad", "value_kw_mean", "id_customer", "temp"]
    df = df[needed].copy()
    df = df[df["type"].isin(["HP", "TOT"])].copy()
    df["dt_utc"] = pd.to_datetime(df["dt_utc"], utc=True, errors="coerce")
    df = df.dropna(subset=["dt_utc"])
    df = convert_f_to_c_if_needed(df)
    df = df.sort_values(["id_customer", "type", "dt_utc"]).reset_index(drop=True)

    modeled_df = build_modeled_dataset(df)
    if modeled_df.empty:
        raise ValueError("No usable customers found in raw training data after filtering")

    # target_name ∈ {"no_hp", "winter_hp", "summer_hp"} → hp_type / hp_label
    modeled_df["hp_type"] = modeled_df["target_name"]
    modeled_df["hp_label"] = modeled_df["target"]
    return modeled_df


class HPDetectorTrainer(AbstractTrainer):
    """Train a Random Forest 3-class classifier for HP presence and season type.

    Labels: 0=no_hp, 1=winter_hp, 2=summer_hp.

    Accepts either:
      - Raw time series training data (all_sources_load_with_weather.parquet format):
        columns [type, source, dt_utc, glob_rad, value_kw_mean, id_customer, temp].
        Feature extraction and label inference run automatically.
      - Pre-computed per-customer feature table with FEATURE_COLS + hp_label (int) or hp_type (str).

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
            logger.info("[HPDetectorTrainer] Raw time series detected — running feature extraction")
            training_df = _build_feature_table_from_raw(training_df)
            logger.info("[HPDetectorTrainer] Feature table: %d customers", len(training_df))

        missing = [c for c in FEATURE_COLS if c not in training_df.columns]
        if missing:
            raise ValueError(f"training_df missing feature columns: {missing}")

        if "hp_label" in training_df.columns:
            y = training_df["hp_label"].astype(int).to_numpy()
        elif "hp_type" in training_df.columns:
            y = training_df["hp_type"].map(LABEL_TO_INT).astype(int).to_numpy()
        else:
            raise ValueError("training_df must have 'hp_label' (int) or 'hp_type' (str) column")

        X = training_df[FEATURE_COLS].to_numpy(dtype=float)

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
            "report": classification_report(
                y_test, y_pred, target_names=["no_hp", "winter_hp", "summer_hp"]
            ),
        }
        self.pipeline_ = pipe
        logger.info("[HPDetectorTrainer] Fitted. F1=%.3f on %d test samples", self._eval_results["f1"], len(X_test))

    def save(self, path: Path) -> None:
        if self.pipeline_ is None:
            raise RuntimeError("Call fit() before save()")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.pipeline_, path)
        logger.info("[HPDetectorTrainer] Model saved → %s", path)

    def evaluate(self) -> dict:
        return dict(self._eval_results)

    @classmethod
    def extract_features_from_timeseries(
        cls,
        customer_df: pd.DataFrame,
        weather_df: pd.DataFrame,
        night_rad_threshold: float = 20.0,
        min_night_rows: int = 100,
    ) -> Optional[dict]:
        """Extract HP features for a single customer. No train/serve skew."""
        df = customer_df.copy()
        df["DT_UTC"] = pd.to_datetime(df["DT_UTC"], errors="coerce")
        df = df.dropna(subset=["DT_UTC"]).sort_values("DT_UTC")

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

        return extract_hp_features(load, temp, rad, ts, night_rad_threshold=night_rad_threshold, min_night_rows=min_night_rows)
