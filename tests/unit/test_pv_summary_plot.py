"""Smoke tests for the matplotlib PV detection summary PNG."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


# matplotlib pulls in a large stack; make tests skip cleanly on a stripped env
matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")  # non-interactive backend for CI

from re_nilm.visualization.pv_summary import (
    plot_pv_capacity_vs_sc_share,
    plot_pv_detection_summary,
    save_pv_capacity_vs_sc_share,
    save_pv_detection_summary,
)


def _synthetic_results(n: int = 500, pv_fraction: float = 0.3) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    n_pv = int(n * pv_fraction)
    has_pv = np.array([True] * n_pv + [False] * (n - n_pv))
    rng.shuffle(has_pv)
    cap = np.where(has_pv, rng.uniform(2, 25, n), np.nan)
    return pd.DataFrame({
        "customer_id": [f"c{i:04d}" for i in range(n)],
        "has_pv": has_pv,
        "pv_capacity_kwp": cap,
        "pv_ci_lower": cap - rng.uniform(0.1, 0.5, n) * np.where(has_pv, 1.0, 0.0),
        "pv_ci_upper": cap + rng.uniform(0.1, 0.5, n) * np.where(has_pv, 1.0, 0.0),
        "sc_share": np.where(has_pv, rng.uniform(0.2, 0.8, n), np.nan),
    })


def test_plot_pv_detection_summary_returns_three_axes():
    fig = plot_pv_detection_summary(_synthetic_results())
    try:
        assert len(fig.axes) == 3, "expected pie + capacity bar + sc bar"
        assert fig._suptitle is not None
        assert "PV" in fig._suptitle.get_text()
    finally:
        import matplotlib.pyplot as plt
        plt.close(fig)


def test_save_pv_detection_summary_writes_png(tmp_path):
    out = tmp_path / "pv_summary.png"
    written = save_pv_detection_summary(_synthetic_results(), out)
    assert written == out
    assert out.exists()
    # PNG magic bytes
    assert out.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    assert out.stat().st_size > 5_000  # not an empty/blank canvas


def test_plot_pv_detection_summary_handles_no_capacity_column():
    """Missing capacity column should annotate the panel, not raise."""
    df = _synthetic_results().drop(columns=["pv_capacity_kwp", "pv_ci_lower", "pv_ci_upper"])
    fig = plot_pv_detection_summary(df)
    import matplotlib.pyplot as plt
    plt.close(fig)


def test_plot_pv_detection_summary_handles_missing_sc():
    df = _synthetic_results().drop(columns=["sc_share"])
    fig = plot_pv_detection_summary(df)
    import matplotlib.pyplot as plt
    plt.close(fig)


def test_plot_pv_detection_summary_requires_has_pv():
    with pytest.raises(ValueError, match="has_pv"):
        plot_pv_detection_summary(pd.DataFrame({"customer_id": ["c0"]}))


def test_plot_pv_capacity_vs_sc_share_renders():
    fig = plot_pv_capacity_vs_sc_share(_synthetic_results())
    try:
        ax = fig.axes[0]
        assert ax.get_yscale() == "linear"
        assert "Self-consumption" in ax.get_ylabel()
        assert "PV capacity" in ax.get_xlabel()
    finally:
        import matplotlib.pyplot as plt
        plt.close(fig)


def test_plot_pv_capacity_vs_sc_share_keeps_zero_rows():
    """sc_share == 0 must still be plotted (at y=0), not silently dropped."""
    df = _synthetic_results()
    df.loc[df["has_pv"], "sc_share"] = 0.0  # everyone has zero self-consumption
    fig = plot_pv_capacity_vs_sc_share(df)
    try:
        scatter = fig.axes[0].collections[0]
        offsets = scatter.get_offsets()
        assert len(offsets) == int(df["has_pv"].sum())
        ys = offsets[:, 1]
        assert (ys == 0.0).all()
    finally:
        import matplotlib.pyplot as plt
        plt.close(fig)


def test_save_pv_capacity_vs_sc_share_writes_png(tmp_path):
    out = tmp_path / "scatter.png"
    written = save_pv_capacity_vs_sc_share(_synthetic_results(), out)
    assert written == out
    assert out.exists()
    assert out.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    assert out.stat().st_size > 5_000


def test_plot_pv_capacity_vs_sc_share_requires_columns():
    with pytest.raises(ValueError, match="pv_capacity_kwp"):
        plot_pv_capacity_vs_sc_share(pd.DataFrame({"has_pv": [True], "sc_share": [0.5]}))
