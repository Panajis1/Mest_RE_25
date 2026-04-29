#!/usr/bin/env python3
"""
Export Plotly figures for the Results section into docs/figures/results/.

Uses prob_summary (CSV or parquet) and optionally pv_indicators_clean parquet
for evaluation dashboard panels. Portfolio capacity panels only need prob_summary.

Example:
  .venv/bin/python scripts/export_results_figures.py
  .venv/bin/python scripts/export_results_figures.py \\
      --prob-summary data/out/capacity_autosave.parquet \\
      --pv-indicators data/out/pv_indicators_clean.parquet \\
      --metadata-dir data/re_data/ETHZ_ALL
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

import matplotlib.pyplot as plt  # noqa: E402

from model.pv_detection import (  # noqa: E402
    aggregate_portfolio_estimates,
    evaluate_portfolio,
    load_customer_metadata,
    plot_evaluation_dashboard,
    plot_portfolio_pv_capacity,
    write_plotly_figures_to_dir,
)


def _load_prob_summary(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    return pd.read_parquet(path)


def _export_portfolio_matplotlib(
    prob_summary: pd.DataFrame,
    portfolio_agg: dict,
    out_dir: Path,
) -> None:
    """Fallback static figures without kaleido (matches key filenames for LaTeX)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    df = prob_summary.dropna(subset=["pv_capacity_kwp"])
    df = df[df["pv_capacity_kwp"] > 0].copy()

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(df["pv_capacity_kwp"], bins=30, color="teal", edgecolor="white", alpha=0.85)
    ax.axvline(portfolio_agg["capacity_stats"]["mean"], color="black", ls="--", label="Mean")
    ax.axvline(portfolio_agg["capacity_stats"]["median"], color="grey", ls=":", label="Median")
    ax.set_xlabel("Estimated PV capacity (kWp)")
    ax.set_ylabel("Customers")
    ax.set_title("Distribution of individual PV capacity estimates")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "capacity_histogram.png", dpi=200)
    plt.close(fig)

    th = portfolio_agg["total_hybrid_kwp"]
    ci_con = portfolio_agg["ci_conservative"]
    err_plus = ci_con[1] - th
    err_minus = th - ci_con[0]

    rc = {
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica Neue", "Arial", "DejaVu Sans"],
        "axes.facecolor": "white",
        "figure.facecolor": "white",
        "axes.edgecolor": "#404040",
        "axes.linewidth": 0.8,
        "axes.grid": False,
        "grid.color": "#E0E0E0",
        "grid.linewidth": 0.8,
        "xtick.major.size": 0,
    }
    with plt.rc_context(rc):
        fig, ax = plt.subplots(figsize=(5.2, 4.2))
        ax.bar(
            [0],
            [th],
            width=0.42,
            color="#2E7D32",
            edgecolor="white",
            linewidth=1.0,
            zorder=2,
        )
        ax.errorbar(
            [0],
            [th],
            yerr=[[err_minus], [err_plus]],
            fmt="none",
            ecolor="#1B5E20",
            capsize=7,
            capthick=2.2,
            elinewidth=2.2,
            zorder=3,
        )
        ax.set_xticks([0])
        ax.set_xticklabels(["Hybrid forecast"])
        ax.set_ylabel("Total capacity (kWp)")
        ax.set_title(
            f"Total portfolio PV capacity: {th:,.0f} kWp\n"
            f"Robust 95% CI [{ci_con[0]:,.0f}, {ci_con[1]:,.0f}] kWp",
            fontsize=11,
            pad=12,
        )
        ax.yaxis.grid(True, which="major", linestyle="-", linewidth=0.8)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ymax = max(ci_con[1], th) * 1.08
        ax.set_ylim(0, ymax)
        ax.bar_label(
            ax.containers[0],
            labels=[f"{th:,.0f} kWp"],
            padding=4,
            fontsize=10,
            fontweight="medium",
        )
        fig.tight_layout()
        fig.savefig(out_dir / "total_capacity_bar.png", dpi=200)
    plt.close(fig)

    agg_sc = portfolio_agg.get("aggregate_sc_share")
    if agg_sc is None or (isinstance(agg_sc, float) and pd.isna(agg_sc)):
        return
    fig, ax = plt.subplots(figsize=(4, 4.5))
    ax.bar([0], [agg_sc], width=0.45, color="mediumpurple", edgecolor="white")
    ax.set_xticks([0])
    ax.set_xticklabels(["Self-consumption share"])
    ax.set_ylim(0, min(1.0, float(agg_sc) + 0.2))
    ax.set_ylabel("Share")
    ax.set_title(f"Aggregate self-consumption (capacity-weighted): {agg_sc:.1%}")
    fig.tight_layout()
    fig.savefig(out_dir / "sc_share_bar.png", dpi=200)
    plt.close(fig)

    agg_sc_nz = portfolio_agg.get("aggregate_sc_share_nonzero_sc")
    if agg_sc_nz is not None and not (isinstance(agg_sc_nz, float) and pd.isna(agg_sc_nz)):
        n_nz = portfolio_agg.get("n_customers_nonzero_sc", 0)
        n_all = portfolio_agg.get("n_customers", 0)
        fig, ax = plt.subplots(figsize=(4, 4.5))
        ax.bar([0], [agg_sc_nz], width=0.45, color="mediumpurple", edgecolor="white")
        ax.set_xticks([0])
        ax.set_xticklabels(["Self-consumption share"])
        ax.set_ylim(0, min(1.0, float(agg_sc_nz) + 0.2))
        ax.set_ylabel("Share")
        ax.set_title(
            f"Aggregate self-consumption (capacity-weighted): {agg_sc_nz:.1%}\n"
            f"Excluding zero SC ({n_nz:,} of {n_all:,} hybrid-PV customers)"
        )
        fig.tight_layout()
        fig.savefig(out_dir / "sc_share_bar_nonzero_sc.png", dpi=200)
        plt.close(fig)


def _export_evaluation_matplotlib(evaluation: dict, out_dir: Path) -> None:
    """Yield histogram + floor vs regression scatter (same basenames as Plotly export)."""
    yield_df = evaluation.get("yield_df")
    if yield_df is None or yield_df.empty:
        return
    sy = yield_df.dropna(subset=["specific_yield_kwh_kwp"])
    if not sy.empty:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        for flag, color in [
            ("plausible", "seagreen"),
            ("low", "orange"),
            ("high", "crimson"),
            ("no_data", "grey"),
        ]:
            m = sy["yield_flag"] == flag
            if m.any():
                ax.hist(
                    sy.loc[m, "specific_yield_kwh_kwp"],
                    bins=30,
                    alpha=0.65,
                    label=flag,
                    color=color,
                )
        ax.axvspan(800, 1200, color="green", alpha=0.08, label="Reference band")
        ax.set_xlabel("Specific yield (kWh/kWp/yr)")
        ax.set_ylabel("Customers")
        ax.set_title("Specific yield distribution (internal screening)")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(out_dir / "yield_distribution.png", dpi=200)
        plt.close(fig)

    valid = yield_df[
        (yield_df["pv_capacity_kwp_regression_only"] > 0)
        & (yield_df["pv_capacity_kwp_floor"] > 0)
    ]
    if valid.empty:
        return
    fig, ax = plt.subplots(figsize=(6, 6))
    for flag, color in [
        ("plausible", "seagreen"),
        ("low", "orange"),
        ("high", "crimson"),
        ("no_data", "grey"),
    ]:
        m = valid["yield_flag"] == flag
        if m.any():
            ax.scatter(
                valid.loc[m, "pv_capacity_kwp_regression_only"],
                valid.loc[m, "pv_capacity_kwp_floor"],
                s=8,
                alpha=0.5,
                label=flag,
                color=color,
            )
    hi = max(
        valid["pv_capacity_kwp_regression_only"].quantile(0.99),
        valid["pv_capacity_kwp_floor"].quantile(0.99),
    )
    ax.plot([0, hi], [0, hi], color="grey", ls="--", label="1:1")
    ax.set_xlabel("Regression capacity (kWp)")
    ax.set_ylabel("Floor capacity (kWp)")
    ax.set_title("Estimator agreement (floor vs regression)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "estimator_agreement.png", dpi=200)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=_repo_root / "docs" / "figures" / "results",
        help="Directory for PNG exports (created if missing).",
    )
    ap.add_argument(
        "--prob-summary",
        type=Path,
        default=_repo_root / "scripts" / "prob_summary_from_autosave.csv",
        help="prob_summary CSV or parquet.",
    )
    ap.add_argument(
        "--pv-indicators",
        type=Path,
        default=_repo_root / "data" / "out" / "pv_indicators_clean.parquet",
        help="Optional pv_indicators_clean parquet (for evaluation dashboard).",
    )
    ap.add_argument(
        "--metadata-dir",
        type=Path,
        default=_repo_root / "data" / "re_data" / "ETHZ_ALL",
        help="Data directory containing metadata parquet (for segment evaluation).",
    )
    ap.add_argument(
        "--matplotlib-only",
        action="store_true",
        help="Skip Plotly/kaleido and write portfolio PNGs with matplotlib only.",
    )
    args = ap.parse_args()

    if not args.prob_summary.exists():
        raise SystemExit(f"Missing prob_summary: {args.prob_summary}")

    prob_summary = _load_prob_summary(args.prob_summary)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    portfolio_agg = aggregate_portfolio_estimates(prob_summary, None)
    if not portfolio_agg:
        raise SystemExit("aggregate_portfolio_estimates returned empty (check prob_summary).")

    if args.matplotlib_only:
        _export_portfolio_matplotlib(prob_summary, portfolio_agg, args.out_dir)
        print(f"Wrote matplotlib portfolio figures to {args.out_dir.resolve()}")
    else:
        try:
            plot_portfolio_pv_capacity(
                prob_summary,
                portfolio_agg,
                show=False,
                save_dir=str(args.out_dir),
            )
            print(f"Wrote portfolio capacity figures to {args.out_dir.resolve()}")
        except RuntimeError as e:
            print(f"Plotly export failed ({e}); using matplotlib fallback.", file=sys.stderr)
            _export_portfolio_matplotlib(prob_summary, portfolio_agg, args.out_dir)
            print(f"Wrote matplotlib portfolio figures to {args.out_dir.resolve()}")

    if args.pv_indicators.exists():
        pv_indicators_clean = pd.read_parquet(args.pv_indicators)
        metadata = load_customer_metadata(str(args.metadata_dir))
        evaluation = evaluate_portfolio(
            prob_summary,
            pv_indicators_clean,
            metadata=metadata,
            segment_col="TYPE_PARTENAIRE_LIBELLE",
        )
        if args.matplotlib_only:
            _export_evaluation_matplotlib(evaluation, args.out_dir)
            print(f"Wrote matplotlib evaluation figures to {args.out_dir.resolve()}")
        else:
            try:
                figs = plot_evaluation_dashboard(evaluation, show=False)
                write_plotly_figures_to_dir(figs, args.out_dir)
                print(f"Wrote evaluation dashboard figures to {args.out_dir.resolve()}")
            except RuntimeError as e:
                print(
                    f"Plotly dashboard export failed ({e}); using matplotlib fallback.",
                    file=sys.stderr,
                )
                _export_evaluation_matplotlib(evaluation, args.out_dir)
                print(f"Wrote matplotlib evaluation figures to {args.out_dir.resolve()}")
    else:
        print(
            f"Skipping evaluation dashboard (no indicators file): {args.pv_indicators}",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
