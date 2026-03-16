#%%
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _get_repo_root() -> Path:
    """
    Resolve the repository root based on this file location or CWD.
    """
    try:
        notebook_file = __file__
    except NameError:
        notebook_file = os.getcwd()
    return Path(notebook_file).resolve().parent.parent


def load_pv_context():
    """
    Reuse the pv_detection loading utilities to get Romande Energie data.
    """
    repo_root = _get_repo_root()
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    from model.pv_detection import load_re_data

    _, re_data_df_small, customer_summary = load_re_data()
    return repo_root, re_data_df_small, customer_summary


def compute_yearly_metrics(customer_summary: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-customer yearly production, consumption and ratio.
    """
    df = customer_summary.copy()
    # Standardise column names from load_re_data() summary
    # Expected columns: ID, sum_cons_kwh, sum_prod_kwh, dt_start, dt_end
    rename_map = {}
    if "sum_cons_kwh" in df.columns:
        rename_map["sum_cons_kwh"] = "yearly_cons_kwh"
    if "sum_prod_kwh" in df.columns:
        rename_map["sum_prod_kwh"] = "yearly_prod_kwh"
    df = df.rename(columns=rename_map)

    # Defensive: drop rows without an ID
    df = df.dropna(subset=["ID"]).copy()
    df["customer_id"] = df["ID"].astype(str)

    # Production / consumption ratio, guard against zero or tiny consumption
    cons = df["yearly_cons_kwh"].astype(float)
    prod = df["yearly_prod_kwh"].astype(float)
    eps = 1e-6
    ratio = np.where(cons > eps, prod / cons, np.nan)
    df["prod_cons_ratio"] = ratio

    return df[
        ["customer_id", "yearly_prod_kwh", "yearly_cons_kwh", "prod_cons_ratio"]
    ].copy()


def _ensure_output_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def plot_distribution(series: pd.Series, title: str, out_path: Path) -> None:
    """
    Save a simple histogram for the given series.
    """
    clean = series.replace([np.inf, -np.inf], np.nan).dropna()
    if clean.empty:
        return

    plt.figure(figsize=(6, 4))
    plt.hist(clean, bins=40, edgecolor="black", alpha=0.7)
    plt.title(title)
    plt.xlabel(series.name)
    plt.ylabel("Number of customers")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def summarise_series(series: pd.Series) -> dict:
    """
    Basic summary statistics for a numeric series.
    """
    clean = series.replace([np.inf, -np.inf], np.nan).dropna()
    if clean.empty:
        return {
            "count": 0,
            "mean": np.nan,
            "median": np.nan,
            "p10": np.nan,
            "p50": np.nan,
            "p90": np.nan,
            "min": np.nan,
            "max": np.nan,
        }

    return {
        "count": int(clean.shape[0]),
        "mean": float(clean.mean()),
        "median": float(clean.median()),
        "p10": float(clean.quantile(0.10)),
        "p50": float(clean.quantile(0.50)),
        "p90": float(clean.quantile(0.90)),
        "min": float(clean.min()),
        "max": float(clean.max()),
    }

#%%
def load_metadata(repo_root: Path) -> pd.DataFrame | None:
    """
    Try to load the Romande Energie metadata table.
    """
    meta_path = repo_root / "data" / "re_data" / "ETHZ" / "metadata"
    if not meta_path.exists():
        return None

    # First try Parquet (as requested), then fall back to CSV with
    # a few reasonable delimiters if that fails.
    try:
        return pd.read_parquet(meta_path)
    except Exception:
        pass

    for loader in ("csv_semicolon", "csv_comma", "csv_tab"):
        try:
            if loader == "csv_semicolon":
                return pd.read_csv(meta_path, sep=";", engine="python")
            if loader == "csv_comma":
                return pd.read_csv(meta_path, sep=",", engine="python")
            if loader == "csv_tab":
                return pd.read_csv(meta_path, sep="\t", engine="python")
        except Exception:
            continue

    return None


def summarise_metadata(meta: pd.DataFrame | None, max_cols: int = 8) -> dict:
    """
    Produce a lightweight summary of the metadata table.
    """
    if meta is None or meta.empty:
        return {"available": False, "columns": [], "examples": {}}

    summary: dict[str, object] = {
        "available": True,
        "columns": list(meta.columns),
        "examples": {},
    }

    # For up to max_cols object-like columns, capture the top few categories
    object_cols = [c for c in meta.columns if meta[c].dtype == "object"]
    for col in object_cols[:max_cols]:
        vc = meta[col].value_counts(dropna=True).head(5)
        summary["examples"][col] = vc.to_dict()

    return summary

#%%
def build_markdown_report(
    metrics_df: pd.DataFrame,
    prod_stats: dict,
    cons_stats: dict,
    ratio_stats: dict,
    metadata_summary: dict,
    plot_dir: Path,
) -> str:
    """
    Construct a short markdown report as a string.
    """
    n_customers = metrics_df.shape[0]

    md_lines: list[str] = []
    md_lines.append("# Romande Energie – Portfolio Summary")
    md_lines.append("")
    md_lines.append(
        f"Number of customers in portfolio (after filters in `load_re_data`): **{n_customers}**."
    )
    md_lines.append("")

    md_lines.append("## Yearly production and consumption")
    md_lines.append("")
    md_lines.append(
        "- **Yearly production (kWh)**: "
        f"mean ≈ {prod_stats['mean']:.1f}, median ≈ {prod_stats['median']:.1f}, "
        f"10–90% range ≈ [{prod_stats['p10']:.1f}, {prod_stats['p90']:.1f}]."
    )
    md_lines.append(
        "- **Yearly consumption (kWh)**: "
        f"mean ≈ {cons_stats['mean']:.1f}, median ≈ {cons_stats['median']:.1f}, "
        f"10–90% range ≈ [{cons_stats['p10']:.1f}, {cons_stats['p90']:.1f}]."
    )
    md_lines.append("")
    md_lines.append(
        f"Histogram plots are saved under `{plot_dir.relative_to(_get_repo_root())}` "
        "(yearly production, yearly consumption)."
    )
    md_lines.append("")

    md_lines.append("## Production-to-consumption ratio")
    md_lines.append("")
    md_lines.append(
        f"- **Ratio distribution** (yearly production / yearly consumption, "
        "excluding customers with ~zero consumption): "
        f"median ≈ {ratio_stats['median']:.3f}, "
        f"10–90% range ≈ [{ratio_stats['p10']:.3f}, {ratio_stats['p90']:.3f}]."
    )
    md_lines.append(
        "- Customers with zero or extremely small yearly consumption are excluded from the ratio distribution but "
        "still counted in the total customer population."
    )
    md_lines.append("")
    md_lines.append(
        f"The ratio histogram is saved as `production_to_consumption_ratio.png` in `{plot_dir.relative_to(_get_repo_root())}`."
    )
    md_lines.append("")

    md_lines.append("## Metadata overview")
    md_lines.append("")
    if not metadata_summary.get("available", False):
        md_lines.append(
            "A metadata file was expected at `data/re_data/ETHZ/metadata`, but it could not be loaded "
            "in a recognised format."
        )
    else:
        cols = metadata_summary.get("columns", [])
        md_lines.append(
            f"- **Columns present** in the metadata table ({len(cols)} total): "
            + ", ".join(str(c) for c in cols)
        )
        examples = metadata_summary.get("examples", {}) or {}
        if examples:
            md_lines.append(
                "- **Example value distributions** for a few categorical columns:"
            )
            for col, values in examples.items():
                pretty_vals = ", ".join(f"`{k}` ({v})" for k, v in values.items())
                md_lines.append(f"  - `{col}`: {pretty_vals}")
        else:
            md_lines.append(
                "No obvious categorical columns were detected for quick value summaries."
            )

    md_lines.append("")
    return "\n".join(md_lines)


def main() -> None:
    repo_root, _, customer_summary = load_pv_context()

    # Yearly metrics
    metrics_df = compute_yearly_metrics(customer_summary)

    # Output directory for plots
    plot_dir = repo_root / "analysis_plots"
    _ensure_output_dir(plot_dir)

    # Distributions and stats
    prod_series = metrics_df["yearly_prod_kwh"]
    cons_series = metrics_df["yearly_cons_kwh"]
    ratio_series = metrics_df["prod_cons_ratio"]

    prod_stats = summarise_series(prod_series)
    cons_stats = summarise_series(cons_series)
    ratio_stats = summarise_series(ratio_series)

    plot_distribution(
        prod_series,
        "Yearly PV production per customer (kWh)",
        plot_dir / "yearly_production_distribution.png",
    )
    plot_distribution(
        cons_series,
        "Yearly consumption per customer (kWh)",
        plot_dir / "yearly_consumption_distribution.png",
    )
    plot_distribution(
        ratio_series,
        "Yearly production / consumption ratio",
        plot_dir / "production_to_consumption_ratio.png",
    )

    # Metadata
    meta_df = load_metadata(repo_root)
    metadata_summary = summarise_metadata(meta_df)

    # Markdown report
    report_md = build_markdown_report(
        metrics_df,
        prod_stats,
        cons_stats,
        ratio_stats,
        metadata_summary,
        plot_dir,
    )
    docs_dir = repo_root / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)
    report_path = docs_dir / "romande_energy_portfolio_report.md"
    report_path.write_text(report_md, encoding="utf-8")


if __name__ == "__main__":
    main()


# %%
