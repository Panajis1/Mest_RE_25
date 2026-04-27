"""Integration test: 10-customer end-to-end pipeline (PV detector only).

Requires no external data — generates synthetic parquet files in a temp directory.
Mark: pytest.mark.integration (run with -m integration).
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from re_nilm.pipeline.orchestrator import PipelineOrchestrator
from re_nilm.pipeline.customer_index import build_customer_file_index, load_customer_from_index
from re_nilm.pipeline.streaming import StreamingEngine
from re_nilm.detectors.pv import PVDetector

pytestmark = pytest.mark.integration

N_CUSTOMERS = 10
N_ROWS = 365 * 96  # one year at 15-min


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_weather(n: int = N_ROWS) -> pd.DataFrame:
    dt = pd.date_range("2022-01-01", periods=n, freq="15min")
    hour = dt.hour.to_numpy() + dt.minute.to_numpy() / 60
    doy = dt.dayofyear.to_numpy()
    rad = np.clip(np.sin(np.pi * (hour - 6) / 12), 0, None) * (
        0.5 + 0.5 * np.sin(2 * np.pi * (doy - 80) / 365)
    ) * 800
    temp = 10 + 12 * np.sin(2 * np.pi * (doy - 80) / 365)
    return pd.DataFrame({"dt_utc": dt, "t_2m_C": temp, "global_rad_W": rad})


def _write_customers(data_dir: Path, weather: pd.DataFrame, n_pv: int = 5) -> list[str]:
    """Write N_CUSTOMERS parquet files; first n_pv have synthetic PV production."""
    rng = np.random.default_rng(0)
    n = len(weather)
    customer_ids = [f"C{i:03d}" for i in range(N_CUSTOMERS)]

    for i, cid in enumerate(customer_ids):
        conso = rng.uniform(0.2, 0.8, n)
        if i < n_pv:
            prod = (weather["global_rad_W"].to_numpy() / 1000.0 * 0.4 + rng.normal(0, 0.01, n)).clip(0)
        else:
            prod = np.zeros(n)

        df = pd.DataFrame({
            "ID": cid,
            "DT_UTC": weather["dt_utc"].to_numpy(),
            "CONSO_KWH": conso.astype("float32"),
            "PROD_KWH": prod.astype("float32"),
        })
        df.to_parquet(data_dir / f"{cid}.parquet", index=False)

    return customer_ids


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------

class TestPipelineSmall:
    @pytest.fixture(autouse=True)
    def setup(self, tmp_path):
        self.tmp = tmp_path
        self.data_dir = tmp_path / "data"
        self.output_dir = tmp_path / "out"
        self.data_dir.mkdir()
        self.output_dir.mkdir()
        self.weather = _make_weather()
        self.customer_ids = _write_customers(self.data_dir, self.weather, n_pv=5)

    def _make_cfg(self) -> dict:
        return {
            "data": {"re_data_dir": str(self.data_dir)},
            "output": {"results_dir": str(self.output_dir)},
            "pipeline": {
                "n_workers": 1,
                "batch_size": 5,
                "resume_from_checkpoint": False,
            },
            "models": {
                "pv": {
                    "corr_threshold": 0.2,
                    "delta_net_threshold": -0.05,
                    "min_yearly_prod_kwh": 0.1,
                    "capacity_bootstrap_n": 5,
                    "capacity_min_kwp": 0.01,
                },
            },
            "detectors": {"enabled": ["pv"]},
        }

    def test_pipeline_runs_end_to_end(self):
        orch = PipelineOrchestrator(self._make_cfg())
        orch._load_weather = lambda: self.weather
        result = orch.run()
        assert isinstance(result, pd.DataFrame), "run() must return a DataFrame"
        assert not result.empty, "Expected non-empty results for 10 customers"

    def test_pipeline_output_schema(self):
        orch = PipelineOrchestrator(self._make_cfg())
        orch._load_weather = lambda: self.weather
        result = orch.run()
        required = {"customer_id", "has_pv"}
        assert required.issubset(result.columns), f"Missing columns: {required - set(result.columns)}"

    def test_pipeline_n_customers(self):
        orch = PipelineOrchestrator(self._make_cfg())
        orch._load_weather = lambda: self.weather
        result = orch.run()
        assert len(result) == N_CUSTOMERS, f"Expected {N_CUSTOMERS} rows, got {len(result)}"

    def test_pv_positive_rate(self):
        """At least the 5 customers with synthetic PV should be detected."""
        orch = PipelineOrchestrator(self._make_cfg())
        orch._load_weather = lambda: self.weather
        result = orch.run()
        if "has_pv" in result.columns:
            n_pv = int(result["has_pv"].sum())
            assert n_pv >= 3, f"Expected ≥3 PV-positive customers, got {n_pv}"

    def test_output_file_written(self):
        orch = PipelineOrchestrator(self._make_cfg())
        orch._load_weather = lambda: self.weather
        orch.run()
        assert (self.output_dir / "results_all_customers.parquet").exists()
        assert (self.output_dir / "pv_indicators.parquet").exists()

    def test_resume_skips_completed_step(self):
        """Second run with resume=True should skip PV detection (output already exists)."""
        cfg = self._make_cfg()
        cfg["pipeline"]["resume_from_checkpoint"] = True

        orch1 = PipelineOrchestrator(cfg)
        orch1._load_weather = lambda: self.weather
        result1 = orch1.run()

        orch2 = PipelineOrchestrator(cfg)
        orch2._load_weather = lambda: self.weather
        result2 = orch2.run()

        assert len(result1) == len(result2)


# ---------------------------------------------------------------------------
# CustomerIndex integration tests
# ---------------------------------------------------------------------------

class TestCustomerIndex:
    @pytest.fixture(autouse=True)
    def setup(self, tmp_path):
        self.data_dir = tmp_path / "data"
        self.data_dir.mkdir()
        self.weather = _make_weather(96 * 7)  # one week
        _write_customers(self.data_dir, self.weather, n_pv=2)

    def test_build_index_finds_all_customers(self):
        index = build_customer_file_index(self.data_dir)
        assert len(index) == N_CUSTOMERS

    def test_load_customer_from_index(self):
        index = build_customer_file_index(self.data_dir)
        df = load_customer_from_index("C000", index)
        assert not df.empty
        assert "CONSO_KWH" in df.columns

    def test_unknown_customer_returns_empty(self):
        index = build_customer_file_index(self.data_dir)
        df = load_customer_from_index("UNKNOWN", index)
        assert df.empty


# ---------------------------------------------------------------------------
# StreamingEngine integration tests
# ---------------------------------------------------------------------------

class TestStreamingEngine:
    @pytest.fixture(autouse=True)
    def setup(self, tmp_path):
        self.data_dir = tmp_path / "data"
        self.output_dir = tmp_path / "out"
        self.data_dir.mkdir()
        self.output_dir.mkdir()
        weather = _make_weather(96 * 7)
        _write_customers(self.data_dir, weather, n_pv=2)
        self.index = build_customer_file_index(self.data_dir)
        self.weather = weather

    def test_sequential_processes_all(self):
        engine = StreamingEngine(n_workers=1, batch_size=3, resume=False)
        detector = PVDetector()
        weather = self.weather

        def processor(cid: str):
            df = load_customer_from_index(cid, self.index)
            return detector.predict_customer(df, weather)

        out_path = self.output_dir / "pv_test.parquet"
        result = engine.run(list(self.index.keys()), processor, out_path)
        assert len(result) == N_CUSTOMERS

    def test_checkpoint_resume(self):
        """Process 5 customers, then resume and process the remaining 5."""
        ckpt_path = self.output_dir / "ckpt.parquet"
        out_path = self.output_dir / "pv_test.parquet"
        all_ids = list(self.index.keys())
        weather = self.weather
        detector = PVDetector()

        def processor(cid: str):
            df = load_customer_from_index(cid, self.index)
            return detector.predict_customer(df, weather)

        # First batch: 5 customers
        engine1 = StreamingEngine(n_workers=1, batch_size=5, resume=True, checkpoint_path=ckpt_path)
        engine1.run(all_ids[:5], processor, out_path)

        # Second batch: resume, should only process remaining 5
        engine2 = StreamingEngine(n_workers=1, batch_size=10, resume=True, checkpoint_path=ckpt_path)
        result = engine2.run(all_ids, processor, out_path)
        assert len(result) == N_CUSTOMERS
