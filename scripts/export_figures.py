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
    docs/figures/results/pv_detection_summary.png         # 3-panel matplotlib PNG
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
from re_nilm.visualization.pv_summary import (
    save_pv_capacity_vs_sc_share,
    save_pv_detection_summary,
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

    # Standalone matplotlib PV plots — written directly because they aren't
    # plotly Figures. Kept independent so a missing matplotlib import doesn't
    # break the rest of the export pipeline.
    if "has_pv" in results.columns:
        try:
            out = save_pv_detection_summary(
                results,
                figures_dir / "pv_detection_summary.png",
            )
            logger.info("PV detection summary written → %s", out)
        except Exception as exc:
            logger.warning("pv_detection_summary failed: %s", exc)

        if "pv_capacity_kwp" in results.columns and "sc_share" in results.columns:
            try:
                out = save_pv_capacity_vs_sc_share(
                    results,
                    figures_dir / "pv_capacity_vs_sc_share.png",
                )
                logger.info("PV capacity-vs-sc scatter written → %s", out)
            except Exception as exc:
                logger.warning("pv_capacity_vs_sc_share failed: %s", exc)

    logger.info("%d figures written to %s", len(figures), figures_dir)
    print(f"\nExported {len(figures)} figures → {figures_dir}")


if __name__ == "__main__":
    main()
