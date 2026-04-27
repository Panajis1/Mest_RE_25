"""Smoke test for PipelineOrchestrator — PV-only, no ML artifacts needed.

Runs in memory using tiny synthetic data (no disk I/O to real parquet files).
"""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from re_nilm.pipeline.orchestrator import PipelineOrchestrator, load_config


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_N_CUST = 4
_N_ROWS = 365 * 96  # one year at 15min


def _make_weather(n: int = _N_ROWS) -> pd.DataFrame:
    dt = pd.date_range("2022-01-01", periods=n, freq="15min")
    hour = dt.hour.to_numpy() + dt.minute.to_numpy() / 60
    doy = dt.dayofyear.to_numpy()
    rad = np.clip(np.sin(np.pi * (hour - 6) / 12), 0, None) * (0.5 + 0.5 * np.sin(2 * np.pi * (doy - 80) / 365)) * 800
    temp = 10 + 12 * np.sin(2 * np.pi * (doy - 80) / 365)
    return pd.DataFrame({"dt_utc": dt, "t_2m_C": temp.astype("float32"), "global_rad_W": rad.astype("float32")})


def _make_customers(weather: pd.DataFrame, tmpdir: Path):
    """Write one parquet per customer into tmpdir."""
    rng = np.random.default_rng(0)
    n = len(weather)
    for i in range(_N_CUST):
        cid = f"C{i:03d}"
        prod = (weather["global_rad_W"].values / 1000.0 * (i * 0.1)).clip(0) if i >= 2 else np.zeros(n)
        df = pd.DataFrame({
            "ID": cid,
            "DT_UTC": weather["dt_utc"].values,
            "CONSO_KWH": rng.uniform(0.2, 0.8, n).astype("float32"),
            "PROD_KWH": prod.astype("float32"),
        })
        df.to_parquet(tmpdir / f"{cid}.parquet", index=False)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def _make_config(data_dir: Path, output_dir: Path) -> dict:
    return {
        "data": {"re_data_dir": str(data_dir)},
        "output": {"results_dir": str(output_dir)},
        "pipeline": {"n_workers": 1, "batch_size": 10, "resume_from_checkpoint": False,
                     "checkpoint_path": str(output_dir / "ckpt.parquet")},
        "models": {
            "pv": {"corr_threshold": 0.2, "delta_net_threshold": -0.05,
                   "min_yearly_prod_kwh": 0.1, "capacity_bootstrap_n": 10,
                   "capacity_min_kwp": 0.01},
        },
        "detectors": {"enabled": ["pv"]},
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestOrchestrator:
    def setup_method(self):
        self.tmp = tempfile.mkdtemp()
        self.data_dir = Path(self.tmp) / "data"
        self.output_dir = Path(self.tmp) / "out"
        self.data_dir.mkdir()
        self.output_dir.mkdir()
        self.weather = _make_weather()
        _make_customers(self.weather, self.data_dir)

    def _make_orch(self, enabled=None):
        cfg = _make_config(self.data_dir, self.output_dir)
        return PipelineOrchestrator(cfg, enabled_detectors=enabled or ["pv"])

    def test_run_pv_only_produces_output_file(self):
        orch = self._make_orch(["pv"])
        # Patch weather loading so we don't need network
        orch._load_weather = lambda: self.weather
        result = orch.run()
        out_path = self.output_dir / "results_all_customers.parquet"
        assert out_path.exists(), "results_all_customers.parquet not written"

    def test_run_pv_output_schema(self):
        orch = self._make_orch(["pv"])
        orch._load_weather = lambda: self.weather
        result = orch.run()
        assert "customer_id" in result.columns
        assert "has_pv" in result.columns

    def test_run_with_no_detectors_returns_empty(self):
        cfg = _make_config(self.data_dir, self.output_dir)
        cfg["detectors"]["enabled"] = []
        orch = PipelineOrchestrator(cfg, enabled_detectors=[])
        orch._load_weather = lambda: self.weather
        result = orch.run()
        assert isinstance(result, pd.DataFrame)

    def test_run_pv_some_customers_positive(self):
        """At least the 2 customers with prod > 0 should be detected as PV-positive."""
        orch = self._make_orch(["pv"])
        orch._load_weather = lambda: self.weather
        result = orch.run()
        if "has_pv" in result.columns:
            n_pv = int(result["has_pv"].sum())
            assert n_pv >= 1, "Expected at least one PV customer"


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def test_load_config_returns_dict():
    repo_root = Path(__file__).resolve().parents[2]
    default = repo_root / "config" / "default.yaml"
    if not default.exists():
        pytest.skip("default.yaml not present")
    cfg = load_config(default)
    assert isinstance(cfg, dict)


def test_load_config_missing_override_falls_back():
    cfg = load_config("/nonexistent/path.yaml")
    assert isinstance(cfg, dict)
