"""HP disaggregation trainer — fits ScientificTwoStageHPModel from labeled TOT+HP time series.

Mirrors the legacy training in `old_files/hp_model/disaggregation_functions.py`
but lives inside the re_nilm package so it can be invoked from
`scripts/train_models.py` without sys.path tricks. Trains only on customers
labeled `winter_hp` (per legacy behaviour) and saves feature_cols on the model
object as well as a JSON sidecar.
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

from re_nilm.estimators._hp_disagg_v1 import (
    EPS,
    HP_ON_THRESHOLD_KW,
    ScientificTwoStageHPModel,
    _add_calendar_and_dynamic_features,
    _compute_user_profile,
)
from re_nilm.training._dataset_builders import (
    convert_f_to_c_if_needed,
    hp_tot_curve_is_usable,
    infer_hp_season_label,
)
from re_nilm.training.base import AbstractTrainer

logger = logging.getLogger(__name__)

_RANDOM_STATE = 42
_PV_FRAC_THRESHOLD = 0.03    # Fraction of negative-load (export) rows that flags a PV customer.
_PV_POWER_THRESHOLD = -0.1   # kW: anything below this is considered an export reading.

_REQUIRED_RAW_COLS = (
    "type", "source", "dt_utc", "glob_rad", "value_kw_mean", "id_customer", "temp",
)
_NON_FEATURE_COLS = {
    "id_customer", "dt_utc", "source_name", "hp_season_label", "profile_class",
    "hp_kw", "hp_ratio", "hp_on", "sample_weight",
}


def _has_strong_pv_signal(tot_df: pd.DataFrame) -> bool:
    """Heuristic: flag a customer as having PV when >=3% of TOT readings export.

    Net export = load < -0.1 kW. PV-heavy customers are excluded from training
    because the TOT curve no longer reflects pure building demand.
    """
    vals = pd.to_numeric(tot_df["value_kw_mean"], errors="coerce").dropna()
    if vals.empty:
        return False
    return float((vals < _PV_POWER_THRESHOLD).mean()) >= _PV_FRAC_THRESHOLD


def _build_aligned_hp_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Merge per-customer TOT and HP sub-meter streams into one feature-rich table.

    Matches the legacy filter chain:
      1. drop customers with strong PV export (TOT no longer = building load),
      2. drop customers whose TOT curve fails hp_tot_curve_is_usable,
      3. keep only customers labeled winter_hp by infer_hp_season_label.
    """
    records = []
    for user_id, user_df in df.groupby("id_customer", sort=False):
        tot_df = user_df[user_df["type"] == "TOT"].copy()
        hp_df = user_df[user_df["type"] == "HP"].copy()
        if tot_df.empty or hp_df.empty:
            continue
        if _has_strong_pv_signal(tot_df):
            continue

        tot_ok, _, _ = hp_tot_curve_is_usable(tot_df)
        if not tot_ok:
            continue

        hp_label, _, _ = infer_hp_season_label(hp_df)
        if hp_label != "winter_hp":
            continue

        source_values = sorted(user_df["source"].dropna().astype(str).unique().tolist())
        source_name = source_values[0] if source_values else "UNKNOWN"

        tot_df = (
            tot_df[["id_customer", "dt_utc", "value_kw_mean", "temp", "glob_rad"]]
            .drop_duplicates(subset=["dt_utc"], keep="first")
            .rename(columns={"value_kw_mean": "tot_kw"})
            .sort_values("dt_utc")
        )
        hp_df = (
            hp_df[["id_customer", "dt_utc", "value_kw_mean"]]
            .drop_duplicates(subset=["dt_utc"], keep="first")
            .rename(columns={"value_kw_mean": "hp_kw"})
            .sort_values("dt_utc")
        )

        merged = tot_df.merge(hp_df, on=["id_customer", "dt_utc"], how="inner")
        merged = merged.loc[merged["tot_kw"].abs() > EPS].reset_index(drop=True)
        if merged.empty:
            continue

        merged["source_name"] = source_name
        merged["hp_season_label"] = hp_label

        profile = _compute_user_profile(merged[["tot_kw", "glob_rad"]].copy())
        for k, v in profile.items():
            merged[k] = v

        merged["hp_ratio"] = (merged["hp_kw"] / (merged["tot_kw"] + EPS)).clip(0.0, 1.0)
        merged["hp_on"] = (merged["hp_kw"] >= HP_ON_THRESHOLD_KW).astype(int)
        merged = _add_calendar_and_dynamic_features(merged)
        records.append(merged)

    if not records:
        raise ValueError("No aligned TOT+HP winter_hp users available after filtering.")

    out = pd.concat(records, axis=0, ignore_index=True)
    logger.info(
        "[HPDisaggTrainer] aligned: %d rows | %d users",
        len(out), out["id_customer"].nunique(),
    )
    return out


def _split_by_user(
    aligned_df: pd.DataFrame, test_size: float
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    users = aligned_df["id_customer"].unique()
    train_u, test_u = train_test_split(users, test_size=test_size, random_state=_RANDOM_STATE)
    train_df = aligned_df[aligned_df["id_customer"].isin(train_u)].copy()
    test_df = aligned_df[aligned_df["id_customer"].isin(test_u)].copy()
    return train_df, test_df


def _add_sample_weights(df: pd.DataFrame) -> pd.DataFrame:
    counts = df["id_customer"].value_counts()
    w = df["id_customer"].map(lambda u: 1.0 / float(counts[u]))
    df = df.copy()
    df["sample_weight"] = (w / w.mean()).astype("float32")
    return df


class HPDisaggregatorTrainer(AbstractTrainer):
    """Train a two-stage HP disaggregation model (classifier + regressor).

    Mirrors the AC disaggregator trainer but uses ScientificTwoStageHPModel
    and the HP-specific dataset filters (winter_hp only, no strong-PV users).

    Args:
        test_size: Fraction of *users* held out for evaluation.
        feature_cols: Optional override; defaults to all columns except
            id_customer/dt_utc/source_name/hp_season_label/profile_class/targets.
    """

    def __init__(
        self,
        test_size: float = 0.2,
        feature_cols: Optional[List[str]] = None,
    ):
        self.test_size = test_size
        self.feature_cols_override = feature_cols
        self.model_: Optional[ScientificTwoStageHPModel] = None
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
        df = df[df["type"].isin(["HP", "TOT"])]
        df["dt_utc"] = pd.to_datetime(df["dt_utc"], utc=True, errors="coerce")
        df = df.dropna(subset=["dt_utc"])
        df = convert_f_to_c_if_needed(df)
        df = df.sort_values(["id_customer", "type", "dt_utc"]).reset_index(drop=True)

        aligned = _build_aligned_hp_dataset(df)
        train_df, test_df = _split_by_user(aligned, test_size=self.test_size)
        train_df = _add_sample_weights(train_df)

        feature_cols = self.feature_cols_override or [
            c for c in aligned.columns if c not in _NON_FEATURE_COLS
        ]
        logger.info("[HPDisaggTrainer] %d feature columns", len(feature_cols))

        model = ScientificTwoStageHPModel(random_state=_RANDOM_STATE)
        model.fit(train_df, feature_cols)

        pred = model.predict(test_df)
        y_on_true = test_df["hp_on"].astype(int).to_numpy()
        y_on_pred = (pred["hp_on_prob"].to_numpy() >= 0.5).astype(int)
        f1 = float(f1_score(y_on_true, y_on_pred, average="weighted"))
        mae_kw = float(mean_absolute_error(test_df["hp_kw"].astype(float), pred["hp_kw_pred"]))

        self.model_ = model
        self.feature_cols_ = list(feature_cols)
        self._eval_results = {
            "n_train_users": int(train_df["id_customer"].nunique()),
            "n_test_users": int(test_df["id_customer"].nunique()),
            "n_train_rows": int(len(train_df)),
            "n_test_rows": int(len(test_df)),
            "f1_hp_on": f1,
            "mae_hp_kw": mae_kw,
        }
        logger.info(
            "[HPDisaggTrainer] fitted: F1(hp_on)=%.3f | MAE(hp_kw)=%.3f kW | "
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
        sidecar = path.with_name(path.stem + "_features.json")
        with open(sidecar, "w") as f:
            json.dump(self.feature_cols_, f, indent=2)
        logger.info("[HPDisaggTrainer] saved → %s (+%s)", path, sidecar.name)

    def evaluate(self) -> dict:
        return dict(self._eval_results)
