"""AC disaggregation trainer — fits AcTwoStageModel from labeled TOT+AC time series.

Invoked from `scripts/train_models.py`. Saves both the model and a feature_cols
JSON sidecar.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, mean_absolute_error
from sklearn.model_selection import train_test_split

from re_nilm.estimators._ac_disagg_v1 import (
    AC_ON_THRESHOLD_KW,
    AcTwoStageModel,
    EPS,
    _add_calendar_and_dynamic_features,
    _compute_user_profile,
)
from re_nilm.training._dataset_builders import (
    ac_tot_curve_is_usable,
    convert_f_to_c_if_needed,
)
from re_nilm.training.base import AbstractTrainer

logger = logging.getLogger(__name__)

_RANDOM_STATE = 42
_REQUIRED_RAW_COLS = (
    "type", "source", "dt_utc", "glob_rad", "value_kw_mean", "id_customer", "temp",
)
_NON_FEATURE_COLS = {
    "id_customer", "dt_utc", "source_name",
    "ac_kw", "ac_ratio", "ac_on", "sample_weight",
}


def _build_aligned_ac_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Merge per-customer TOT and AC sub-meter streams into one feature-rich table.

    Drops customers whose TOT curve fails ac_tot_curve_is_usable (gaps, missing
    summer rows, etc.). Adds user profile, calendar/dynamic features, and the
    targets ac_kw / ac_on / ac_ratio.
    """
    records = []
    for user_id, user_df in df.groupby("id_customer", sort=False):
        tot_df = user_df[user_df["type"] == "TOT"].copy()
        ac_df = user_df[user_df["type"] == "AC"].copy()
        if tot_df.empty or ac_df.empty:
            continue

        usable, _, _ = ac_tot_curve_is_usable(tot_df)
        if not usable:
            continue

        source = (
            str(user_df["source"].dropna().iloc[0])
            if not user_df["source"].dropna().empty
            else "UNKNOWN"
        )

        tot_df = (
            tot_df[["id_customer", "dt_utc", "value_kw_mean", "temp", "glob_rad"]]
            .drop_duplicates("dt_utc", keep="first")
            .rename(columns={"value_kw_mean": "tot_kw"})
            .sort_values("dt_utc")
        )
        ac_df = (
            ac_df[["dt_utc", "value_kw_mean"]]
            .drop_duplicates("dt_utc", keep="first")
            .rename(columns={"value_kw_mean": "ac_kw"})
            .sort_values("dt_utc")
        )

        merged = tot_df.merge(ac_df, on="dt_utc", how="inner")
        merged = merged[merged["tot_kw"].abs() > EPS].reset_index(drop=True)
        if merged.empty:
            continue

        merged["source_name"] = source
        profile = _compute_user_profile(merged[["tot_kw", "glob_rad"]])
        for k, v in profile.items():
            merged[k] = v

        merged["ac_ratio"] = (merged["ac_kw"] / (merged["tot_kw"] + EPS)).clip(0.0, 1.0)
        merged["ac_on"] = (merged["ac_kw"] >= AC_ON_THRESHOLD_KW).astype(int)
        merged = _add_calendar_and_dynamic_features(merged)

        for col in merged.select_dtypes("float64").columns:
            merged[col] = merged[col].astype("float32")

        records.append(merged)

    if not records:
        raise ValueError("No aligned TOT+AC users found after curve-usability filtering.")

    out = pd.concat(records, ignore_index=True)
    logger.info(
        "[ACDisaggTrainer] aligned: %d rows | %d users",
        len(out), out["id_customer"].nunique(),
    )
    return out


def _split_by_user(
    aligned_df: pd.DataFrame, test_size: float
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Hold out entire users — never split rows from the same customer across train/test."""
    users = aligned_df["id_customer"].unique()
    train_u, test_u = train_test_split(users, test_size=test_size, random_state=_RANDOM_STATE)
    train_df = aligned_df[aligned_df["id_customer"].isin(train_u)].copy()
    test_df = aligned_df[aligned_df["id_customer"].isin(test_u)].copy()
    return train_df, test_df


def _add_sample_weights(df: pd.DataFrame) -> pd.DataFrame:
    """Per-row weight = 1 / N_rows_for_customer (then mean-normalised)."""
    counts = df["id_customer"].value_counts()
    w = df["id_customer"].map(lambda u: 1.0 / float(counts[u]))
    df = df.copy()
    df["sample_weight"] = (w / w.mean()).astype("float32")
    return df


class ACDisaggregatorTrainer(AbstractTrainer):
    """Train a two-stage AC disaggregation model (classifier + regressor).

    Accepts the same raw all_sources_load_with_weather.parquet format used by
    the AC detector trainer. Splits by user (not by row) so evaluation is on
    held-out customers, then fits AcTwoStageModel and writes a feature_cols
    JSON sidecar alongside the model artifact.

    Args:
        test_size: Fraction of *users* to hold out for evaluation.
        feature_cols: Optional override; defaults to all columns except
            id_customer/dt_utc/source_name/targets/sample_weight.
    """

    def __init__(
        self,
        test_size: float = 0.2,
        feature_cols: Optional[List[str]] = None,
    ):
        self.test_size = test_size
        self.feature_cols_override = feature_cols
        self.model_: Optional[AcTwoStageModel] = None
        self.feature_cols_: Optional[List[str]] = None
        self._eval_results: dict = {}

    def fit(self, training_df: pd.DataFrame) -> None:
        missing = [c for c in _REQUIRED_RAW_COLS if c not in training_df.columns]
        if missing:
            raise ValueError(
                f"training_df missing required raw columns {missing}. "
                f"Expected the all_sources format with columns {list(_REQUIRED_RAW_COLS)}."
            )

        df = training_df[list(_REQUIRED_RAW_COLS)].copy()
        df = df[df["type"].isin(["AC", "TOT"])]
        df["dt_utc"] = pd.to_datetime(df["dt_utc"], utc=True, errors="coerce")
        df = df.dropna(subset=["dt_utc"])
        df = convert_f_to_c_if_needed(df)
        df = df.sort_values(["id_customer", "type", "dt_utc"]).reset_index(drop=True)

        aligned = _build_aligned_ac_dataset(df)
        train_df, test_df = _split_by_user(aligned, test_size=self.test_size)
        train_df = _add_sample_weights(train_df)

        feature_cols = self.feature_cols_override or [
            c for c in aligned.columns if c not in _NON_FEATURE_COLS
        ]
        logger.info("[ACDisaggTrainer] %d feature columns", len(feature_cols))

        model = AcTwoStageModel(random_state=_RANDOM_STATE)
        model.fit(train_df, feature_cols)

        # Evaluate on held-out users.
        pred = model.predict(test_df, apply_month_gating=True)
        y_on_true = test_df["ac_on"].astype(int).to_numpy()
        y_on_pred = (pred["ac_on_prob"].to_numpy() >= 0.5).astype(int)
        f1 = float(f1_score(y_on_true, y_on_pred, average="weighted"))
        mae_kw = float(mean_absolute_error(test_df["ac_kw"].astype(float), pred["ac_kw_pred"]))

        self.model_ = model
        self.feature_cols_ = list(feature_cols)
        self._eval_results = {
            "n_train_users": int(train_df["id_customer"].nunique()),
            "n_test_users": int(test_df["id_customer"].nunique()),
            "n_train_rows": int(len(train_df)),
            "n_test_rows": int(len(test_df)),
            "f1_ac_on": f1,
            "mae_ac_kw": mae_kw,
        }
        logger.info(
            "[ACDisaggTrainer] fitted: F1(ac_on)=%.3f | MAE(ac_kw)=%.3f kW | "
            "%d train users / %d test users",
            f1, mae_kw,
            self._eval_results["n_train_users"], self._eval_results["n_test_users"],
        )

    def save(self, path: Path) -> None:
        if self.model_ is None or self.feature_cols_ is None:
            raise RuntimeError("Call fit() before save().")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model_, path)
        # Sidecar JSON with feature columns — matches the existing
        # ac_disaggregator_v1_features.json convention.
        sidecar = path.with_name(path.stem + "_features.json")
        with open(sidecar, "w") as f:
            json.dump(self.feature_cols_, f, indent=2)
        logger.info("[ACDisaggTrainer] saved → %s (+%s)", path, sidecar.name)

    def evaluate(self) -> dict:
        return dict(self._eval_results)
