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
    """

    def __init__(
        self,
        config: Dict[str, Any],
        enabled_detectors: Optional[List[str]] = None,
        resume: Optional[bool] = None,
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

        self.data_dir = Path(self._data_cfg.get("re_data_dir", "data/raw/re"))
        self.output_dir = Path(self._output_cfg.get("results_dir", "data/processed/out"))

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

        streaming_cfg = dict(
            n_workers=self._pipe_cfg.get("n_workers", 1),
            batch_size=self._pipe_cfg.get("batch_size", 500),
            resume=self.resume,
        )

        pv_result = self._run_pv(customer_ids, index, weather_df, streaming_cfg)
        cap_result = self._run_pv_capacity(customer_ids, index, weather_df, pv_result, streaming_cfg)
        batt_result = self._run_battery(customer_ids, index, weather_df, cap_result, streaming_cfg)
        ac_result = self._run_ac(customer_ids, index, weather_df, streaming_cfg)
        hp_result = self._run_hp(customer_ids, index, weather_df, streaming_cfg)
        ev_result = self._run_ev(customer_ids, index, weather_df, streaming_cfg)

        self._run_ac_disagg(customer_ids, index, weather_df, ac_result)
        self._run_hp_disagg(customer_ids, index, weather_df, hp_result)
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
        weather_cache = self.output_dir / "weather_cache.parquet"
        loader = WeatherLoader(backend="meteoswiss", cache_dir=self.output_dir)
        try:
            return loader.load()
        except Exception as exc:
            logger.warning("[Orchestrator] Weather load failed (%s) — weather columns will be NaN", exc)
            return pd.DataFrame(columns=["dt_utc", "t_2m_C", "global_rad_W"])

    def _build_index(self) -> Dict[str, List[str]]:
        cache_path = self.output_dir / "customer_file_index.json"
        return build_customer_file_index(self.data_dir, cache_path=cache_path)

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

        def processor(cid: str):
            df = load_customer_from_index(cid, index)
            return detector.predict_customer(df, weather_df)

        engine = self._make_engine(out_path, streaming_cfg)
        result = engine.run(customer_ids, processor, out_path)
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

        def processor(cid: str):
            det = pv_lookup.get(cid, {"customer_id": cid, "has_pv": False})
            if not det.get("has_pv", False):
                return None
            df = load_customer_from_index(cid, index)
            return estimator.estimate(df, det, weather_df)

        engine = self._make_engine(out_path, streaming_cfg)
        result = engine.run(list(pv_pos), processor, out_path)
        logger.info("[Orchestrator] PV capacity: %d estimates", len(result))
        return result

    def _run_battery(self, customer_ids, index, weather_df, cap_result, streaming_cfg) -> pd.DataFrame:
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

        cap_lookup = cap_result.set_index("customer_id").to_dict("index") if not cap_result.empty else {}

        def processor(cid: str):
            pv_cap = cap_lookup.get(cid, {})
            df = load_customer_from_index(cid, index)
            det = detector.predict_customer(df, weather_df, pv_result=pv_cap)
            if det is None:
                return None
            cap_est = estimator.estimate(df, {**det, "pv_result": pv_cap}, weather_df)
            combined = dict(det)
            if cap_est:
                combined.update({k: v for k, v in cap_est.items() if k != "customer_id"})
            return combined

        engine = self._make_engine(out_path, streaming_cfg)
        result = engine.run(customer_ids, processor, out_path)
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
            inference_months=ac_cfg.get("inference_months", [6, 7, 8]),
        )

        def processor(cid: str):
            df = load_customer_from_index(cid, index)
            return detector.predict_customer(df, weather_df)

        engine = self._make_engine(out_path, streaming_cfg)
        result = engine.run(customer_ids, processor, out_path)
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
            prob_threshold=hp_cfg.get("prob_threshold", 0.5),
            night_rad_threshold=hp_cfg.get("night_rad_threshold", 20.0),
        )

        def processor(cid: str):
            df = load_customer_from_index(cid, index)
            return detector.predict_customer(df, weather_df)

        engine = self._make_engine(out_path, streaming_cfg)
        result = engine.run(customer_ids, processor, out_path)
        logger.info("[Orchestrator] HP detection: %d customers, %d positive",
                    len(result), int(result["has_hp"].sum()) if "has_hp" in result.columns else 0)
        return result

    def _run_ac_disagg(self, customer_ids, index, weather_df, ac_result: pd.DataFrame):
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

        all_frames: List[pd.DataFrame] = []
        for cid in ac_lookup:
            df = load_customer_from_index(cid, index)
            result = estimator.estimate(df, ac_lookup[cid], weather_df)
            if result and "ac_disagg_15min" in result:
                all_frames.append(result["ac_disagg_15min"])

        if all_frames:
            out = pd.concat(all_frames, ignore_index=True)
            out.to_parquet(out_path, index=False)
            logger.info("[Orchestrator] AC disagg written: %d rows → %s", len(out), out_path)

    def _run_hp_disagg(self, customer_ids, index, weather_df, hp_result: pd.DataFrame):
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

        all_frames: List[pd.DataFrame] = []
        for cid in hp_lookup:
            df = load_customer_from_index(cid, index)
            result = estimator.estimate(df, hp_lookup[cid], weather_df)
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

        def processor(cid: str):
            df = load_customer_from_index(cid, index)
            return detector.predict_customer(df, weather_df)

        engine = self._make_engine(out_path, streaming_cfg)
        result = engine.run(customer_ids, processor, out_path)
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
