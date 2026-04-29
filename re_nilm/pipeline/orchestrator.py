"""Pipeline orchestrator — composes detectors and estimators into a single run.

Execution order (mirrors ticklish-weaving-crown.md Step 7):
  1. Load weather once.
  2. Build / load customer-file index.
  3. PV Detector → pv_indicators.parquet
  4. PV Capacity Estimator → pv_capacity.parquet
  5. Battery Detector + Estimator (needs PV) → battery_results.parquet
  6. AC Detector (needs trained model) → ac_detection.parquet
  7. HP Detector (needs trained model) → hp_detection.parquet
  8. AC Disaggregator (AC-positive only) → ac_disagg_15min.parquet
  9. HP Disaggregator (winter_hp only)  → hp_disagg_15min.parquet
 10. Join scalar results → results_all_customers.parquet

ML detectors/disaggregators are skipped with a warning when their model
artifacts are missing — the pipeline can always run PV and battery.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional

import pandas as pd
import yaml

from re_nilm.pipeline.customer_index import build_customer_file_index, load_customer_from_index
from re_nilm.data.loaders.smart_meter import SmartMeterLoader
from re_nilm.data.loaders.weather import WeatherLoader
from re_nilm.detectors.battery import BatteryDetector
from re_nilm.detectors.pv import PVDetector
from re_nilm.estimators.battery_capacity import BatteryCapacityEstimator
from re_nilm.estimators.pv_capacity import PVCapacityEstimator
from re_nilm.pipeline.streaming import StreamingEngine

logger = logging.getLogger(__name__)
_REPO_ROOT = Path(__file__).resolve().parents[2]


def _resolve_repo_path(path_like: str | Path) -> Path:
    """Resolve config paths relative to repository root."""
    p = Path(path_like)
    return p if p.is_absolute() else (_REPO_ROOT / p).resolve()


# ---------------------------------------------------------------------------
# Per-worker state (multiprocessing support)
#
# ProcessPoolExecutor pickles each pool.submit(fn, arg) call independently.
# Embedding large objects (index dict, weather DataFrame, sklearn model) inside
# a closure or per-call callable means they get pickled once per customer —
# O(n_customers × state_size) of serialisation overhead per batch.
#
# The initializer pattern avoids this: state is pickled once per *worker
# process* at pool creation time, stored in a module-level dict, and accessed
# by the thin module-level worker functions below.  In sequential mode
# (n_workers <= 1) StreamingEngine calls the initializer once before the loop.
# ---------------------------------------------------------------------------

_WORKER_STATE: Dict[str, Any] = {}


def _init_worker(state: Dict[str, Any]) -> None:
    """Initialise per-process worker state. Called once per worker by ProcessPoolExecutor."""
    global _WORKER_STATE
    _WORKER_STATE = state


def _detector_worker(cid: str) -> Optional[dict]:
    """Generic detector worker — load customer data and call predict_customer."""
    s = _WORKER_STATE
    df = load_customer_from_index(cid, s["index"])
    return s["detector"].predict_customer(df, s["weather_df"])


def _pv_capacity_worker(cid: str) -> Optional[dict]:
    """PV capacity worker — skip non-PV customers, run capacity estimator."""
    s = _WORKER_STATE
    det = s["pv_lookup"].get(cid, {"customer_id": cid, "has_pv": False})
    if not det.get("has_pv", False):
        return None
    df = load_customer_from_index(cid, s["index"])
    return s["estimator"].estimate(df, det, s["weather_df"])


def _battery_worker(cid: str) -> Optional[dict]:
    """Battery worker — run detector + capacity estimator, propagating PV result."""
    s = _WORKER_STATE
    pv_det = s["pv_lookup"].get(cid, {})
    pv_cap = s["cap_lookup"].get(cid, {})
    pv_ctx = {**pv_det, **pv_cap}
    df = load_customer_from_index(cid, s["index"])
    det = s["detector"].predict_customer(df, s["weather_df"], pv_result=pv_ctx)
    if det is None:
        return None
    cap_est = s["estimator"].estimate(df, {**det, "pv_result": pv_ctx}, s["weather_df"])
    combined = dict(det)
    if cap_est:
        combined.update({k: v for k, v in cap_est.items() if k != "customer_id"})
    return combined


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def load_config(path: str | Path) -> Dict[str, Any]:
    """Load default.yaml then overlay a production YAML on top."""
    repo_root = Path(__file__).resolve().parents[2]
    default_path = repo_root / "config" / "default.yaml"

    cfg: Dict[str, Any] = {}
    if default_path.exists():
        with open(default_path) as f:
            cfg = yaml.safe_load(f) or {}

    if path is not None:
        override_path = Path(path)
        if override_path.exists():
            with open(override_path) as f:
                override = yaml.safe_load(f) or {}
            cfg = _deep_merge(cfg, override)
        else:
            logger.warning("Config override not found: %s — using defaults only", override_path)
    return cfg


def _deep_merge(base: dict, override: dict) -> dict:
    result = dict(base)
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

class PipelineOrchestrator:
    """Run the full appliance detection pipeline driven by a YAML config.

    Args:
        config: Loaded config dict (from load_config()).
        enabled_detectors: Override which detectors to run. Defaults to config value.
        resume: Resume from checkpoint if True.
        skip_ac_disagg: Skip AC 15-minute disaggregation while preserving scalar results.
        skip_hp_disagg: Skip HP 15-minute disaggregation while preserving scalar results.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        enabled_detectors: Optional[List[str]] = None,
        resume: Optional[bool] = None,
        skip_ac_disagg: Optional[bool] = None,
        skip_hp_disagg: Optional[bool] = None,
    ):
        self.cfg = config
        self.enabled = set(
            enabled_detectors
            if enabled_detectors is not None
            else config.get("detectors", {}).get("enabled", ["pv", "battery", "ac", "heat_pump"])
        )
        self.resume = resume if resume is not None else config.get("pipeline", {}).get("resume_from_checkpoint", True)

        self._data_cfg = config.get("data", {})
        self._pipe_cfg = config.get("pipeline", {})
        self._models_cfg = config.get("models", {})
        self._output_cfg = config.get("output", {})
        self.skip_ac_disagg = (
            skip_ac_disagg
            if skip_ac_disagg is not None
            else bool(self._pipe_cfg.get("skip_ac_disagg", False))
        )
        self.skip_hp_disagg = (
            skip_hp_disagg
            if skip_hp_disagg is not None
            else bool(self._pipe_cfg.get("skip_hp_disagg", False))
        )

        self.data_dir = _resolve_repo_path(self._data_cfg.get("re_data_dir", "data/raw/re"))
        self.output_dir = _resolve_repo_path(self._output_cfg.get("results_dir", "data/processed/out"))

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(self) -> pd.DataFrame:
        """Execute the full pipeline and return the combined scalar results."""
        self.output_dir.mkdir(parents=True, exist_ok=True)

        logger.info("[Orchestrator] Loading weather...")
        weather_df = self._load_weather()

        logger.info("[Orchestrator] Building customer file index...")
        index = self._build_index()
        customer_ids = list(index.keys())
        logger.info("[Orchestrator] %d customers found", len(customer_ids))

        self._check_weather_coverage(weather_df, index)

        raw_workers = self._pipe_cfg.get("n_workers", 1)
        n_workers = os.cpu_count() or 1 if raw_workers == "auto" else int(raw_workers)
        logger.info("[Orchestrator] Using %d workers (cpu_count=%d)", n_workers, os.cpu_count() or 1)

        streaming_cfg = dict(
            n_workers=n_workers,
            batch_size=self._pipe_cfg.get("batch_size", 500),
            resume=self.resume,
        )

        allowed_ids = set(customer_ids)
        pv_result = self._restrict_results_to_customers(
            self._run_pv(customer_ids, index, weather_df, streaming_cfg),
            allowed_ids,
            "PV indicators",
        )
        cap_result = self._restrict_results_to_customers(
            self._run_pv_capacity(customer_ids, index, weather_df, pv_result, streaming_cfg),
            allowed_ids,
            "PV capacity",
        )
        batt_result = self._restrict_results_to_customers(
            self._run_battery(customer_ids, index, weather_df, pv_result, cap_result, streaming_cfg),
            allowed_ids,
            "Battery results",
        )
        ac_result = self._restrict_results_to_customers(
            self._run_ac(customer_ids, index, weather_df, streaming_cfg),
            allowed_ids,
            "AC detection",
        )
        hp_result = self._restrict_results_to_customers(
            self._run_hp(customer_ids, index, weather_df, streaming_cfg),
            allowed_ids,
            "HP detection",
        )
        ev_result = self._restrict_results_to_customers(
            self._run_ev(customer_ids, index, weather_df, streaming_cfg),
            allowed_ids,
            "EV detection",
        )

        if self.skip_ac_disagg:
            logger.info("[Orchestrator] AC disaggregation skipped by config/CLI")
        else:
            self._run_ac_disagg(customer_ids, index, weather_df, ac_result, cap_result)

        if self.skip_hp_disagg:
            logger.info("[Orchestrator] HP disaggregation skipped by config/CLI")
        else:
            self._run_hp_disagg(customer_ids, index, weather_df, hp_result, cap_result)

        self._run_ev_sessions(customer_ids, index, weather_df, ev_result)

        combined = self._join_scalar_results(pv_result, cap_result, batt_result, ac_result, hp_result, ev_result)
        out_path = self.output_dir / "results_all_customers.parquet"
        combined.to_parquet(out_path, index=False)
        logger.info("[Orchestrator] Results written → %s (%d customers)", out_path, len(combined))
        return combined

    # ------------------------------------------------------------------
    # Step helpers
    # ------------------------------------------------------------------

    def _load_weather(self) -> pd.DataFrame:
        weather_required: bool = self._pipe_cfg.get("weather_required", True)
        loader = WeatherLoader(backend="meteoswiss", cache_dir=self.output_dir)
        try:
            return loader.load()
        except Exception as exc:
            if weather_required:
                raise RuntimeError(
                    "Weather load failed and pipeline.weather_required is true. "
                    "Fix weather access or set weather_required: false in config to allow "
                    "degraded mode (AC, HP, battery detectors will return NaN features). "
                    f"Original error: {exc}"
                ) from exc
            logger.warning(
                "[Orchestrator] Weather load failed (%s) — running in degraded mode. "
                "Set pipeline.weather_required: true to make weather failures fatal.",
                exc,
            )
            return pd.DataFrame(columns=["dt_utc", "t_2m_C", "global_rad_W"])

    def _build_index(self) -> Dict[str, List[str]]:
        cache_path = self.output_dir / "customer_file_index.json"
        return build_customer_file_index(
            self.data_dir,
            cache_path=cache_path,
            customer_type_filter=self._data_cfg.get("customer_type_filter", "Particuliers"),
            max_consumption_kwh=self._data_cfg.get("max_consumption_kwh", 100_000),
        )

    def _make_engine(self, output_path: Path, streaming_cfg: dict) -> StreamingEngine:
        # Each pipeline step uses its own checkpoint derived from its output path so
        # steps do not contaminate each other's resume state.
        ckpt = output_path.with_name(output_path.stem + "_ckpt.parquet")
        return StreamingEngine(
            n_workers=streaming_cfg["n_workers"],
            batch_size=streaming_cfg["batch_size"],
            resume=streaming_cfg["resume"],
            checkpoint_path=ckpt,
        )

    def _customer_loader(self, index: Dict, weather_df: pd.DataFrame):
        """Return a processor_fn factory that loads one customer's data."""
        def _load(cid: str):
            return load_customer_from_index(cid, index), weather_df
        return _load

    def _restrict_results_to_customers(
        self,
        df: pd.DataFrame,
        allowed_ids: set[str],
        label: str,
    ) -> pd.DataFrame:
        """Keep resumed result tables aligned with the filtered customer index."""
        if df.empty or "customer_id" not in df.columns:
            return df
        before = len(df)
        out = df[df["customer_id"].astype(str).isin(allowed_ids)].copy()
        if len(out) != before:
            logger.info(
                "[Orchestrator] %s restricted to filtered cohort: %d/%d rows",
                label,
                len(out),
                before,
            )
        return out

    def _check_weather_coverage(self, weather_df: pd.DataFrame, index: Dict) -> None:
        """Warn if weather data does not cover the smart meter date range.

        Samples up to 10 unique parquet files from the index to estimate the
        smart meter min/max date, then compares against weather_df's dt_utc range.
        Logs a WARNING if either end has a gap larger than 30 days.
        """
        if weather_df.empty or not index:
            return

        # Collect up to 10 unique parquet file paths across all customers.
        seen_files: set = set()
        for paths in index.values():
            seen_files.update(paths)
            if len(seen_files) >= 10:
                break

        meter_min: Optional[pd.Timestamp] = None
        meter_max: Optional[pd.Timestamp] = None
        for path in list(seen_files)[:10]:
            try:
                ts = pd.to_datetime(pd.read_parquet(path, columns=["DT_UTC"])["DT_UTC"])
                local_min, local_max = ts.min(), ts.max()
                if meter_min is None or local_min < meter_min:
                    meter_min = local_min
                if meter_max is None or local_max > meter_max:
                    meter_max = local_max
            except Exception:
                continue

        if meter_min is None:
            logger.warning("[Orchestrator] Could not sample smart meter dates — skipping weather coverage check.")
            return

        # Normalise both sides to tz-naive for comparison.
        def _strip_tz(t: pd.Timestamp) -> pd.Timestamp:
            return t.tz_localize(None) if t.tzinfo is not None else t

        meter_min = _strip_tz(meter_min)
        meter_max = _strip_tz(meter_max)

        weather_ts = pd.to_datetime(weather_df["dt_utc"])
        weather_min = _strip_tz(weather_ts.min())
        weather_max = _strip_tz(weather_ts.max())

        gap_start = max(0.0, (weather_min - meter_min).total_seconds() / 86400)
        gap_end = max(0.0, (meter_max - weather_max).total_seconds() / 86400)

        if gap_start > 30 or gap_end > 30:
            logger.warning(
                "[Orchestrator] Weather coverage gap: smart meter spans %s–%s but weather spans "
                "%s–%s (gap: %.0f days at start, %.0f days at end). "
                "Weather-dependent detectors (AC, HP, battery) may produce NaN features.",
                meter_min.date(), meter_max.date(),
                weather_min.date(), weather_max.date(),
                gap_start, gap_end,
            )
        else:
            logger.info(
                "[Orchestrator] Weather coverage OK: weather %s–%s covers smart meter ~%s–%s.",
                weather_min.date(), weather_max.date(),
                meter_min.date(), meter_max.date(),
            )

    def _run_pv(self, customer_ids, index, weather_df, streaming_cfg) -> pd.DataFrame:
        out_path = self.output_dir / "pv_indicators.parquet"
        if "pv" not in self.enabled:
            return pd.DataFrame(columns=["customer_id", "has_pv", "prob_pv"])
        if out_path.exists() and self.resume:
            logger.info("[Orchestrator] PV indicators already exist — skipping")
            return pd.read_parquet(out_path)

        pv_cfg = self._models_cfg.get("pv", {})
        detector = PVDetector(
            corr_threshold=pv_cfg.get("corr_threshold", 0.3),
            delta_net_threshold=pv_cfg.get("delta_net_threshold", -0.1),
            min_yearly_prod_kwh=pv_cfg.get("min_yearly_prod_kwh", 1.0),
        )

        worker_state = {"detector": detector, "index": index, "weather_df": weather_df}
        engine = self._make_engine(out_path, streaming_cfg)
        result = engine.run(
            customer_ids, _detector_worker, out_path,
            worker_initializer=_init_worker, worker_initargs=(worker_state,),
        )
        logger.info("[Orchestrator] PV detection: %d customers, %d PV-positive",
                    len(result), int(result["has_pv"].sum()) if "has_pv" in result.columns else 0)
        return result

    def _run_pv_capacity(self, customer_ids, index, weather_df, pv_result, streaming_cfg) -> pd.DataFrame:
        out_path = self.output_dir / "pv_capacity.parquet"
        if "pv" not in self.enabled:
            return pd.DataFrame(columns=["customer_id", "pv_capacity_kwp"])
        if out_path.exists() and self.resume:
            logger.info("[Orchestrator] PV capacity already exists — skipping")
            return pd.read_parquet(out_path)

        pv_cfg = self._models_cfg.get("pv", {})
        estimator = PVCapacityEstimator(
            bootstrap_n=pv_cfg.get("capacity_bootstrap_n", 200),
            capacity_min_kwp=pv_cfg.get("capacity_min_kwp", 0.1),
        )

        pv_pos = set(pv_result.loc[pv_result.get("has_pv", False) == True, "customer_id"].astype(str)) if "has_pv" in pv_result.columns else set()
        # Use drop=False so customer_id remains in the dict values after set_index
        pv_lookup = pv_result.set_index("customer_id", drop=False).to_dict("index") if not pv_result.empty else {}

        worker_state = {
            "estimator": estimator, "pv_lookup": pv_lookup,
            "index": index, "weather_df": weather_df,
        }
        engine = self._make_engine(out_path, streaming_cfg)
        result = engine.run(
            list(pv_pos), _pv_capacity_worker, out_path,
            worker_initializer=_init_worker, worker_initargs=(worker_state,),
        )
        logger.info("[Orchestrator] PV capacity: %d estimates", len(result))
        return result

    def _run_battery(self, customer_ids, index, weather_df, pv_result, cap_result, streaming_cfg) -> pd.DataFrame:
        out_path = self.output_dir / "battery_results.parquet"
        if "battery" not in self.enabled:
            return pd.DataFrame(columns=["customer_id", "has_battery", "prob_battery"])
        if out_path.exists() and self.resume:
            logger.info("[Orchestrator] Battery results already exist — skipping")
            return pd.read_parquet(out_path)

        batt_cfg = self._models_cfg.get("battery", {})
        detector = BatteryDetector(
            classification_threshold=batt_cfg.get("classification_threshold", 0.5),
            enforce_pv_required=batt_cfg.get("enforce_pv_required", True),
            dark_day_rad_max_w=batt_cfg.get("dark_day_rad_max_w", 100.0),
            sunny_day_rad_min_w=batt_cfg.get("sunny_day_rad_min_w", 100.0),
        )
        estimator = BatteryCapacityEstimator(
            capacity_min_kwh=batt_cfg.get("capacity_min_kwh", 5.0),
            capacity_max_kwh=batt_cfg.get("capacity_max_kwh", 30.0),
        )

        pv_lookup = pv_result.set_index("customer_id", drop=False).to_dict("index") if not pv_result.empty else {}
        cap_lookup = cap_result.set_index("customer_id", drop=False).to_dict("index") if not cap_result.empty else {}

        worker_state = {
            "detector": detector, "estimator": estimator,
            "pv_lookup": pv_lookup, "cap_lookup": cap_lookup,
            "index": index, "weather_df": weather_df,
        }
        engine = self._make_engine(out_path, streaming_cfg)
        result = engine.run(
            customer_ids, _battery_worker, out_path,
            worker_initializer=_init_worker, worker_initargs=(worker_state,),
        )
        logger.info("[Orchestrator] Battery detection: %d customers, %d positive",
                    len(result), int(result.get("has_battery", pd.Series()).sum()) if "has_battery" in result.columns else 0)
        return result

    def _run_ac(self, customer_ids, index, weather_df, streaming_cfg) -> pd.DataFrame:
        out_path = self.output_dir / "ac_detection.parquet"
        if "ac" not in self.enabled:
            return pd.DataFrame(columns=["customer_id", "has_ac", "prob_ac"])
        if out_path.exists() and self.resume:
            logger.info("[Orchestrator] AC detection already exists — skipping")
            return pd.read_parquet(out_path)

        ac_cfg = self._models_cfg.get("ac", {})
        detector_path = Path(ac_cfg.get("detector_path", "models/ac_detector_v1.joblib"))
        if not detector_path.exists():
            logger.warning("[Orchestrator] AC detector not found at %s — skipping AC detection", detector_path)
            return pd.DataFrame(columns=["customer_id", "has_ac", "prob_ac"])

        from re_nilm.detectors.ac import ACDetector
        detector = ACDetector.load(
            detector_path,
            prob_threshold=ac_cfg.get("prob_threshold", 0.55),
            day_rad_threshold=float(ac_cfg.get("day_rad_threshold", 50.0)),
            min_day_rows=int(ac_cfg.get("min_day_rows", 100)),
        )

        worker_state = {"detector": detector, "index": index, "weather_df": weather_df}
        engine = self._make_engine(out_path, streaming_cfg)
        result = engine.run(
            customer_ids, _detector_worker, out_path,
            worker_initializer=_init_worker, worker_initargs=(worker_state,),
        )
        logger.info("[Orchestrator] AC detection: %d customers, %d positive",
                    len(result), int(result["has_ac"].sum()) if "has_ac" in result.columns else 0)
        return result

    def _run_hp(self, customer_ids, index, weather_df, streaming_cfg) -> pd.DataFrame:
        out_path = self.output_dir / "hp_detection.parquet"
        if "heat_pump" not in self.enabled:
            return pd.DataFrame(columns=["customer_id", "has_hp", "prob_hp", "hp_type"])
        if out_path.exists() and self.resume:
            logger.info("[Orchestrator] HP detection already exists — skipping")
            return pd.read_parquet(out_path)

        hp_cfg = self._models_cfg.get("heat_pump", {})
        detector_path = Path(hp_cfg.get("detector_path", "models/hp_detector_v1.joblib"))
        if not detector_path.exists():
            logger.warning("[Orchestrator] HP detector not found at %s — skipping HP detection", detector_path)
            return pd.DataFrame(columns=["customer_id", "has_hp", "prob_hp", "hp_type"])

        from re_nilm.detectors.heat_pump import HeatPumpDetector
        detector = HeatPumpDetector.load(
            detector_path,
            night_rad_threshold=hp_cfg.get("night_rad_threshold", 20.0),
            min_night_rows=int(hp_cfg.get("min_night_rows", 100)),
        )

        worker_state = {"detector": detector, "index": index, "weather_df": weather_df}
        engine = self._make_engine(out_path, streaming_cfg)
        result = engine.run(
            customer_ids, _detector_worker, out_path,
            worker_initializer=_init_worker, worker_initargs=(worker_state,),
        )
        logger.info("[Orchestrator] HP detection: %d customers, %d positive",
                    len(result), int(result["has_hp"].sum()) if "has_hp" in result.columns else 0)
        return result

    def _run_ac_disagg(
        self,
        customer_ids,
        index,
        weather_df,
        ac_result: pd.DataFrame,
        cap_result: pd.DataFrame,
    ):
        out_path = self.output_dir / "ac_disagg_15min.parquet"
        if "ac" not in self.enabled or ac_result.empty or "has_ac" not in ac_result.columns:
            return
        if out_path.exists() and self.resume:
            logger.info("[Orchestrator] AC disaggregation already exists — skipping")
            return

        ac_cfg = self._models_cfg.get("ac", {})
        disagg_path = Path(ac_cfg.get("disaggregator_path", "models/ac_disaggregator_v1.pkl"))
        feat_path = Path(str(disagg_path).replace(".pkl", "_features.json"))
        if not disagg_path.exists():
            logger.warning("[Orchestrator] AC disaggregator not found at %s — skipping", disagg_path)
            return

        from re_nilm.estimators.ac_disaggregation import ACDisaggregationEstimator
        estimator = ACDisaggregationEstimator.load(
            disagg_path,
            features_path=feat_path if feat_path.exists() else None,
        )

        ac_pos = ac_result[ac_result["has_ac"] == True]
        ac_lookup = ac_pos.set_index("customer_id", drop=False).to_dict("index")

        # Build {customer_id → pv_capacity_kwp} for PV correction in _build_features.
        # Customers absent from cap_result (no PV) get 0.0, preserving existing behaviour.
        cap_lookup: Dict[str, float] = {}
        if not cap_result.empty and "customer_id" in cap_result.columns and "pv_capacity_kwp" in cap_result.columns:
            for _, row in cap_result[["customer_id", "pv_capacity_kwp"]].iterrows():
                kwp = row["pv_capacity_kwp"]
                if pd.notna(kwp) and kwp > 0.0:
                    cap_lookup[str(row["customer_id"])] = float(kwp)

        all_frames: List[pd.DataFrame] = []
        for cid in ac_lookup:
            df = load_customer_from_index(cid, index)
            pv_capacity_kwp = cap_lookup.get(str(cid), 0.0)
            result = estimator.estimate(df, ac_lookup[cid], weather_df, pv_capacity_kwp=pv_capacity_kwp)
            if result and "ac_disagg_15min" in result:
                all_frames.append(result["ac_disagg_15min"])

        if all_frames:
            out = pd.concat(all_frames, ignore_index=True)
            out.to_parquet(out_path, index=False)
            logger.info("[Orchestrator] AC disagg written: %d rows → %s", len(out), out_path)

    def _run_hp_disagg(
        self,
        customer_ids,
        index,
        weather_df,
        hp_result: pd.DataFrame,
        cap_result: pd.DataFrame,
    ):
        out_path = self.output_dir / "hp_disagg_15min.parquet"
        if "heat_pump" not in self.enabled or hp_result.empty or "hp_type" not in hp_result.columns:
            return
        if out_path.exists() and self.resume:
            logger.info("[Orchestrator] HP disaggregation already exists — skipping")
            return

        hp_cfg = self._models_cfg.get("heat_pump", {})
        disagg_path = Path(hp_cfg.get("disaggregator_path", "models/hp_disaggregator_v1.joblib"))
        if not disagg_path.exists():
            logger.warning("[Orchestrator] HP disaggregator not found at %s — skipping", disagg_path)
            return

        from re_nilm.estimators.hp_disaggregation import HPDisaggregationEstimator
        estimator = HPDisaggregationEstimator.load(disagg_path)

        winter_hp = hp_result[hp_result["hp_type"] == "winter_hp"]
        hp_lookup = winter_hp.set_index("customer_id", drop=False).to_dict("index")

        # Build {customer_id → pv_capacity_kwp} for PV correction in _build_features.
        # Customers absent from cap_result (no PV) get 0.0, preserving existing behaviour.
        cap_lookup: Dict[str, float] = {}
        if not cap_result.empty and "customer_id" in cap_result.columns and "pv_capacity_kwp" in cap_result.columns:
            for _, row in cap_result[["customer_id", "pv_capacity_kwp"]].iterrows():
                kwp = row["pv_capacity_kwp"]
                if pd.notna(kwp) and kwp > 0.0:
                    cap_lookup[str(row["customer_id"])] = float(kwp)

        all_frames: List[pd.DataFrame] = []
        for cid in hp_lookup:
            df = load_customer_from_index(cid, index)
            pv_capacity_kwp = cap_lookup.get(str(cid), 0.0)
            result = estimator.estimate(df, hp_lookup[cid], weather_df, pv_capacity_kwp=pv_capacity_kwp)
            if result and "hp_disagg_15min" in result:
                all_frames.append(result["hp_disagg_15min"])

        if all_frames:
            out = pd.concat(all_frames, ignore_index=True)
            out.to_parquet(out_path, index=False)
            logger.info("[Orchestrator] HP disagg written: %d rows → %s", len(out), out_path)

    def _run_ev(self, customer_ids, index, weather_df, streaming_cfg) -> pd.DataFrame:
        out_path = self.output_dir / "ev_detection.parquet"
        if "ev" not in self.enabled:
            return pd.DataFrame(columns=["customer_id", "has_ev", "prob_ev"])
        if out_path.exists() and self.resume:
            logger.info("[Orchestrator] EV detection already exists — skipping")
            return pd.read_parquet(out_path)

        from re_nilm.detectors.ev import EVDetector
        ev_cfg = self._models_cfg.get("ev", {})
        detector = EVDetector(
            prob_threshold=ev_cfg.get("prob_threshold", 0.3),
            max_yearly_mwh=ev_cfg.get("max_yearly_mwh", 100.0),
            grid_threshold_kw=ev_cfg.get("grid_threshold_kw", 2.8),
            min_session_steps=ev_cfg.get("min_session_steps", 6),
            max_rel_std=ev_cfg.get("max_rel_std", 0.15),
        )

        worker_state = {"detector": detector, "index": index, "weather_df": weather_df}
        engine = self._make_engine(out_path, streaming_cfg)
        result = engine.run(
            customer_ids, _detector_worker, out_path,
            worker_initializer=_init_worker, worker_initargs=(worker_state,),
        )
        n_pos = int(result["has_ev"].sum()) if "has_ev" in result.columns else 0
        logger.info("[Orchestrator] EV detection: %d customers, %d EV-positive", len(result), n_pos)
        return result

    def _run_ev_sessions(self, customer_ids, index, weather_df, ev_result: pd.DataFrame):
        out_path = self.output_dir / "ev_sessions_15min.parquet"
        if "ev" not in self.enabled or ev_result.empty or "has_ev" not in ev_result.columns:
            return
        if out_path.exists() and self.resume:
            logger.info("[Orchestrator] EV sessions already exist — skipping")
            return

        from re_nilm.estimators.ev_sessions import EVSessionEstimator
        ev_cfg = self._models_cfg.get("ev_sessions", {})
        estimator = EVSessionEstimator(
            step_threshold_kwh=ev_cfg.get("step_threshold_kwh", 1.5),
            min_duration_minutes=ev_cfg.get("min_duration_minutes", 60),
        )

        ev_pos = ev_result[ev_result["has_ev"] == True]
        ev_lookup = ev_pos.set_index("customer_id", drop=False).to_dict("index")

        all_frames: List[pd.DataFrame] = []
        for cid in ev_lookup:
            df = load_customer_from_index(cid, index)
            result = estimator.estimate(df, ev_lookup[cid], weather_df)
            if result and "ev_sessions" in result:
                all_frames.append(result["ev_sessions"])

        if all_frames:
            out = pd.concat(all_frames, ignore_index=True)
            out.to_parquet(out_path, index=False)
            logger.info("[Orchestrator] EV sessions written: %d rows → %s", len(out), out_path)

    def _join_scalar_results(
        self,
        pv: pd.DataFrame,
        cap: pd.DataFrame,
        batt: pd.DataFrame,
        ac: pd.DataFrame,
        hp: pd.DataFrame,
        ev: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """Left-join all scalar result tables on customer_id."""
        frames = [pv, cap, batt, ac, hp, ev] if ev is not None else [pv, cap, batt, ac, hp]
        result = None
        for df in frames:
            if df.empty or "customer_id" not in df.columns:
                continue
            df = df.copy()
            df["customer_id"] = df["customer_id"].astype(str)
            if result is None:
                result = df
            else:
                overlap = [c for c in df.columns if c != "customer_id" and c in result.columns]
                df_clean = df.drop(columns=overlap) if overlap else df
                result = result.merge(df_clean, on="customer_id", how="outer")

        return result if result is not None else pd.DataFrame()
