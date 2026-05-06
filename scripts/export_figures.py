"""Export portfolio result figures from the pipeline output to docs/figures/.

Reads the joined results parquet produced by run_pipeline.py and generates
summary visualisations (population statistics, capacity vs. production).
Figures are written as PNG files.

Prerequisites:
    - Pipeline results at the path set by ``output.results_dir`` in the config
      (default: ``data/processed/out/results_all_customers.parquet``).
      Run ``run_pipeline.py`` first if these files are missing.

Output:
    docs/figures/results/appliance_adoption_shares.png
    docs/figures/results/pv_installed_capacity_summary.png
    docs/figures/results/pv_capacity_distribution.png
    docs/figures/results/appliance_probability_distributions.png
    docs/figures/results/appliance_cooccurrence_heatmap.png
    docs/figures/results/technology_portfolio_summaries.png
    docs/figures/results/pv_population_statistics.png
    docs/figures/results/pv_capacity_vs_production.png

Usage:
    python scripts/export_figures.py --config config/re_production.yaml
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from re_nilm.pipeline.orchestrator import load_config
from re_nilm.visualization.portfolio import (
    plot_appliance_adoption_shares,
    plot_appliance_cooccurrence_heatmap,
    plot_appliance_probability_distributions,
    plot_capacity_vs_production_with_ci,
    plot_population_statistics,
    plot_pv_capacity_distribution,
    plot_pv_installed_capacity_summary,
    plot_technology_portfolio_summaries,
    write_plotly_figures_to_dir,
)


def _parse_args():
    parser = argparse.ArgumentParser(description="Export RE-NILM portfolio figures")
    parser.add_argument("--config", default="config/re_production.yaml")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser.parse_args()


def main():
    args = _parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    logger = logging.getLogger("export_figures")

    cfg = load_config(args.config)
    results_dir = Path(cfg.get("output", {}).get("results_dir", "data/processed/out"))
    figures_dir = Path(cfg.get("output", {}).get("figures_dir", "docs/figures/results"))

    results_path = results_dir / "results_all_customers.parquet"
    if not results_path.exists():
        logger.error("Results not found at %s — run run_pipeline.py first", results_path)
        sys.exit(1)

    import pandas as pd
    results = pd.read_parquet(results_path)
    logger.info("Loaded %d customers from %s", len(results), results_path)

    figures = []

    for name, plotter in [
        ("appliance_adoption_shares", plot_appliance_adoption_shares),
        ("appliance_probability_distributions", plot_appliance_probability_distributions),
        ("appliance_cooccurrence_heatmap", plot_appliance_cooccurrence_heatmap),
        ("technology_portfolio_summaries", plot_technology_portfolio_summaries),
    ]:
        try:
            figures.append((name, plotter(results)))
        except Exception as exc:
            logger.warning("%s failed: %s", name, exc)

    if "pv_capacity_kwp" in results.columns:
        try:
            fig = plot_pv_installed_capacity_summary(results)
            figures.append(("pv_installed_capacity_summary", fig))
        except Exception as exc:
            logger.warning("plot_pv_installed_capacity_summary failed: %s", exc)

        try:
            fig = plot_pv_capacity_distribution(results, x_max_kwp=100.0)
            figures.append(("pv_capacity_distribution", fig))
        except Exception as exc:
            logger.warning("plot_pv_capacity_distribution failed: %s", exc)

        try:
            fig = plot_population_statistics(results)
            figures.append(("pv_population_statistics", fig))
        except Exception as exc:
            logger.warning("plot_population_statistics failed: %s", exc)

        try:
            pv_ind_path = results_dir / "pv_indicators.parquet"
            if pv_ind_path.exists():
                pv_ind = pd.read_parquet(pv_ind_path)
                fig = plot_capacity_vs_production_with_ci(results.merge(pv_ind, on="customer_id", how="left"))
                figures.append(("pv_capacity_vs_production", fig))
        except Exception as exc:
            logger.warning("plot_capacity_vs_production_with_ci failed: %s", exc)

    if not figures:
        logger.warning("No figures generated — check that results contain appliance result columns")
        return

    write_plotly_figures_to_dir(figures, figures_dir, fmt="png")
    logger.info("%d figures written to %s", len(figures), figures_dir)
    print(f"\nExported {len(figures)} figures → {figures_dir}")

    _print_stats(results, results_dir)


def _print_stats(results: pd.DataFrame, results_dir: Path) -> None:
    import numpy as np

    sep = "=" * 60
    print(f"\n{sep}")
    print("  PRESENTATION STATISTICS SUMMARY")
    print(sep)
    n_total = len(results)
    print(f"  Total customers analysed : {n_total:,}")

    appliance_cols = {
        "AC":         "has_ac",
        "Heat Pump":  "has_hp",
        "EV":         "has_ev",
        "PV":         "has_pv",
        "Battery":    "has_battery",
    }
    print(f"\n  {'Appliance':<14} {'Detected':>10} {'Share':>8}  {'95% CI':>16}")
    print(f"  {'-'*14} {'-'*10} {'-'*8}  {'-'*16}")
    for name, col in appliance_cols.items():
        if col not in results.columns:
            continue
        valid = results[col].notna()
        n = int(valid.sum())
        k = int((results.loc[valid, col].astype(bool)).sum())
        p = k / n if n else 0.0
        # Wilson CI
        from scipy import stats as _stats
        z = 1.96
        denom = 1 + z**2 / n
        centre = (p + z**2 / (2*n)) / denom
        margin = z * ((p*(1-p)/n + z**2/(4*n**2))**0.5) / denom
        lo, hi = max(0, centre - margin) * 100, min(100, centre + margin) * 100
        print(f"  {name:<14} {k:>10,} {p*100:>7.1f}%  [{lo:.1f}%–{hi:.1f}%]")

    # AC-specific stats
    if "has_ac" in results.columns and "prob_ac" in results.columns:
        ac = results[results["has_ac"] == True]
        print(f"\n  AC Detection detail")
        print(f"    Mean prob_ac (all)    : {results['prob_ac'].mean():.3f}")
        print(f"    Mean prob_ac (AC+)   : {results.loc[results['has_ac']==True,'prob_ac'].mean():.3f}")

    # AC disaggregation
    disagg_path = results_dir / "ac_disagg_15min.parquet"
    if disagg_path.exists():
        import pandas as pd
        d = pd.read_parquet(disagg_path)
        if "ac_kw_pred" in d.columns:
            per_cust = d.groupby("customer_id")["ac_kw_pred"].mean()
            print(f"\n  AC Disaggregation")
            print(f"    Customers disaggregated : {len(per_cust):,}")
            print(f"    Mean AC load (kW)       : {per_cust.mean():.3f}")
            print(f"    Median AC load (kW)     : {per_cust.median():.3f}")
            print(f"    Mean annual AC (kWh)    : {(per_cust * 8760).mean():.0f}")

    print(f"\n{sep}\n")


if __name__ == "__main__":
    main()
