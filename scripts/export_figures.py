"""Export portfolio result figures to the docs/figures/ directory.

Merges export_results_figures.py and re_portfolio_report.py into a single
config-driven script.

Usage:
    python scripts/export_figures.py --config config/re_production.yaml
    python scripts/export_figures.py --format html
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
    plot_population_statistics,
    plot_capacity_vs_production_with_ci,
    write_plotly_figures_to_dir,
)


def _parse_args():
    parser = argparse.ArgumentParser(description="Export RE-NILM portfolio figures")
    parser.add_argument("--config", default="config/re_production.yaml")
    parser.add_argument("--format", default="png", choices=["png", "html"], dest="fmt")
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

    if "pv_capacity_kwp" in results.columns:
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
        logger.warning("No figures generated — check that results contain PV columns")
        return

    write_plotly_figures_to_dir(figures, figures_dir, fmt=args.fmt)
    logger.info("%d figures written to %s", len(figures), figures_dir)
    print(f"\nExported {len(figures)} figures → {figures_dir}")


if __name__ == "__main__":
    main()
