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
    docs/figures/results/pv_probability_distribution.png
    docs/figures/results/ac_probability_distribution.png
    docs/figures/results/hp_probability_distribution.png
    docs/figures/results/battery_probability_distribution.png
    docs/figures/results/ev_probability_distribution.png
    docs/figures/results/battery_capacity_histogram.png
    docs/figures/results/battery_reliability_summary.png
    docs/figures/results/battery_probability_distribution_matplotlib.png
    docs/figures/results/hp_customer_mix_pie.png
    docs/figures/results/hp_annual_consumption_pdf.png
    docs/figures/results/ev_probability_distribution_hist.png
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
    plot_appliance_probability_distribution,
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
from re_nilm.visualization.battery_summary import (
    save_battery_capacity_boxplot,
    save_battery_capacity_coverage,
    save_battery_capacity_histogram,
    save_battery_dark_days_distribution,
    save_battery_detected_share_pie,
    save_battery_matched_days_distribution,
    save_battery_portfolio_summary_tables,
    save_battery_probability_distribution,
    save_battery_pv_split_pie,
    save_battery_reliability_summary,
    save_battery_status_breakdown,
    save_pv_vs_battery_capacity_scatter,
)
from re_nilm.visualization.hp_ev_summary import (
    save_ev_probability_histogram,
    save_hp_annual_consumption_pdf,
    save_hp_customer_mix_pie,
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

    # If a previous pipeline run skipped PV (or wrote results before the
    # join was extended), the joined parquet may be missing the PV columns
    # while the per-step files still exist. Augment from those step files
    # so the PV plots can use the correct has_pv / pv_capacity / sc_share.
    pv_ind_path = results_dir / "pv_indicators.parquet"
    if pv_ind_path.exists() and "has_pv" not in results.columns:
        pv_ind = pd.read_parquet(pv_ind_path)
        before = len(results.columns)
        results = results.merge(pv_ind, on="customer_id", how="outer")
        logger.info("Merged %s (+%d cols)", pv_ind_path.name, len(results.columns) - before)

    pv_cap_path = results_dir / "pv_capacity.parquet"
    if pv_cap_path.exists() and "pv_capacity_kwp" not in results.columns:
        pv_cap = pd.read_parquet(pv_cap_path)
        before = len(results.columns)
        results = results.merge(pv_cap, on="customer_id", how="outer")
        logger.info("Merged %s (+%d cols)", pv_cap_path.name, len(results.columns) - before)

    figures = []

    for name, plotter in [
        ("appliance_adoption_shares", plot_appliance_adoption_shares),
        ("appliance_cooccurrence_heatmap", plot_appliance_cooccurrence_heatmap),
        ("technology_portfolio_summaries", plot_technology_portfolio_summaries),
    ]:
        try:
            figures.append((name, plotter(results)))
        except Exception as exc:
            logger.warning("%s failed: %s", name, exc)

    for appliance, stem in [
        ("pv", "pv_probability_distribution"),
        ("ac", "ac_probability_distribution"),
        ("hp", "hp_probability_distribution"),
        ("battery", "battery_probability_distribution"),
        ("ev", "ev_probability_distribution"),
    ]:
        try:
            figures.append((stem, plot_appliance_probability_distribution(results, appliance=appliance)))
        except Exception as exc:
            logger.warning("%s failed: %s", stem, exc)

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

    if "has_battery" in results.columns:
        batt_cfg = cfg.get("models", {}).get("battery", {})
        threshold = float(batt_cfg.get("classification_threshold", 0.5))
        reliability_margin = float(batt_cfg.get("reliability_margin", 0.15))
        try:
            out = save_battery_capacity_histogram(
                results,
                figures_dir / "battery_capacity_histogram.png",
            )
            logger.info("Battery capacity histogram written → %s", out)
        except Exception as exc:
            logger.warning("battery_capacity_histogram failed: %s", exc)

        try:
            out = save_battery_reliability_summary(
                results,
                figures_dir / "battery_reliability_summary.png",
                threshold=threshold,
                reliability_margin=reliability_margin,
            )
            logger.info("Battery reliability summary written → %s", out)
        except Exception as exc:
            logger.warning("battery_reliability_summary failed: %s", exc)

        try:
            out = save_battery_probability_distribution(
                results,
                figures_dir / "battery_probability_distribution_matplotlib.png",
                threshold=threshold,
            )
            logger.info("Battery probability distribution written → %s", out)
        except Exception as exc:
            logger.warning("battery_probability_distribution_matplotlib failed: %s", exc)

        for stem, fn, kw in [
            ("battery_detected_share_pie", save_battery_detected_share_pie, {"threshold": threshold}),
            ("battery_status_breakdown", save_battery_status_breakdown, {}),
            ("battery_pv_split_pie", save_battery_pv_split_pie, {}),
            ("battery_capacity_boxplot", save_battery_capacity_boxplot, {}),
            ("battery_capacity_coverage", save_battery_capacity_coverage, {}),
            ("battery_matched_days_distribution", save_battery_matched_days_distribution, {}),
            ("battery_dark_days_distribution", save_battery_dark_days_distribution, {}),
            ("pv_vs_battery_capacity_scatter", save_pv_vs_battery_capacity_scatter, {}),
        ]:
            try:
                out = fn(results, figures_dir / f"{stem}.png", **kw)
                logger.info("%s written → %s", stem, out)
            except Exception as exc:
                logger.warning("%s failed: %s", stem, exc)

        try:
            paths = save_battery_portfolio_summary_tables(
                results,
                figures_dir,
                threshold=threshold,
                reliability_margin=reliability_margin,
            )
            for label, p in paths.items():
                logger.info("battery_portfolio_%s_table written → %s", label, p)
        except Exception as exc:
            logger.warning("battery_portfolio_summary_tables failed: %s", exc)

    if "has_hp" in results.columns or "hp_type" in results.columns:
        try:
            out = save_hp_customer_mix_pie(
                results,
                figures_dir / "hp_customer_mix_pie.png",
            )
            logger.info("HP customer mix pie written → %s", out)
        except Exception as exc:
            logger.warning("hp_customer_mix_pie failed: %s", exc)

        # Prefer the joined scalar column when present (cheap). Fall back to a
        # streamed pyarrow aggregation over hp_disagg_15min.parquet — that file
        # can be multi-GB on full portfolio runs, so a plain pd.read_parquet
        # spikes RAM and risks OOM/laptop-crash. iter_batches keeps peak
        # memory bounded to one batch (~200k rows ≈ tens of MB).
        hp_annual = None
        if "hp_annual_kwh" in results.columns:
            hp_annual = pd.to_numeric(results["hp_annual_kwh"], errors="coerce")
        else:
            hp_disagg_path = results_dir / "hp_disagg_15min.parquet"
            if hp_disagg_path.exists():
                try:
                    import pyarrow.parquet as pq
                    sums: dict[str, float] = {}
                    pf = pq.ParquetFile(hp_disagg_path)
                    for batch in pf.iter_batches(
                        batch_size=200_000,
                        columns=["customer_id", "hp_kw_pred"],
                    ):
                        bdf = batch.to_pandas()
                        s = (
                            pd.to_numeric(bdf["hp_kw_pred"], errors="coerce")
                            .groupby(bdf["customer_id"].astype(str))
                            .sum(min_count=1)
                        )
                        for cid, val in s.items():
                            if pd.notna(val):
                                sums[cid] = sums.get(cid, 0.0) + float(val)
                        del bdf, s
                    hp_annual = pd.Series(sums, dtype=float) * 0.25
                except Exception as exc:
                    logger.warning("Failed streaming hp_disagg_15min.parquet for annual HP PDF: %s", exc)

        if hp_annual is not None:
            try:
                out = save_hp_annual_consumption_pdf(
                    hp_annual,
                    figures_dir / "hp_annual_consumption_pdf.png",
                )
                logger.info("HP annual consumption PDF written → %s", out)
            except Exception as exc:
                logger.warning("hp_annual_consumption_pdf failed: %s", exc)

    if "prob_ev" in results.columns:
        try:
            out = save_ev_probability_histogram(
                results,
                figures_dir / "ev_probability_distribution_hist.png",
            )
            logger.info("EV probability histogram written → %s", out)
        except Exception as exc:
            logger.warning("ev_probability_distribution_hist failed: %s", exc)

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
