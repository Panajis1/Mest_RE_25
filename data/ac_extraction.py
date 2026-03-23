"""
AC Load Extraction from Total Load Curve.

Uses Dataport households with ground-truth AC metering to train a regression model
that estimates AC load from total load + weather/time features. Can then be applied
to sources that only have total load (e.g. RE/Romande Energie).

Run from repo root:
  python scripts/ac_load_extraction.py              # train + predict on unified data
  python scripts/ac_load_extraction.py --re-data-dir "C:\\Users\\jiniy\\Desktop\\CS\\ETHZ"  # also predict on RE
"""
# %%
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

REPO_ROOT = _repo_root
PROCESSED_DIR = REPO_ROOT / "data" / "processed_data"
UNIFIED_PATH = PROCESSED_DIR / "all_sources_load_with_weather.parquet"
MODEL_DIR = REPO_ROOT / "model"
# Default RE data path (many parquet files); override via --re-data-dir
DEFAULT_RE_DATA_DIR = r"C:\Users\jiniy\Desktop\CS\ETHZ"
MODEL_DIR.mkdir(parents=True, exist_ok=True)


def load_and_pivot(df: pd.DataFrame) -> pd.DataFrame:
    """Pivot long format to wide: one row per (source, id_customer, dt_utc) with TOT, AC columns."""
    pivoted = df.pivot_table(
        index=["source", "id_customer", "dt_utc"],
        columns="type",
        values="value_kw_mean",
        aggfunc="first",
    ).reset_index()
    return pivoted


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create features for AC load prediction."""
    df = df.copy()
    df["dt_utc"] = pd.to_datetime(df["dt_utc"], utc=True)
    df["hour"] = df["dt_utc"].dt.hour
    df["dow"] = df["dt_utc"].dt.dayofweek
    df["month"] = df["dt_utc"].dt.month
    return df


def load_dataport_only() -> pd.DataFrame:
    """Load only Dataport rows from unified parquet (avoids loading 35M rows)."""
    import pyarrow.parquet as pq

    table = pq.read_table(UNIFIED_PATH, filters=[("source", "==", "dataport")])
    return table.to_pandas()


def prepare_training_data(df_dataport: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    Prepare X (features) and y (AC load) for model training.
    Uses only Dataport rows that have both TOT and AC.
    """
    pivoted = load_and_pivot(df_dataport)
    pivoted = build_features(pivoted)
    train_df = pivoted.copy()

    if "TOT" not in train_df.columns or "AC" not in train_df.columns:
        raise ValueError(
            "Dataport data must contain both TOT and AC types. "
            "Check that dataport_appliance_households.parquet has AC households."
        )

    # Merge weather (per source)
    weather = (
        df_dataport[["source", "dt_utc", "temp", "glob_rad"]]
        .drop_duplicates(subset=["source", "dt_utc"])
    )
    train_df = train_df.merge(weather, on=["source", "dt_utc"], how="left")

    # Drop rows with missing target or key features
    train_df = train_df.dropna(subset=["TOT", "AC", "temp"])
    train_df["glob_rad"] = train_df["glob_rad"].fillna(0)

    feature_cols = ["TOT", "temp", "glob_rad", "hour", "dow", "month"]
    X = train_df[feature_cols]
    y = train_df["AC"]

    return X, y, train_df


def train_and_evaluate(
    X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, random_state: int = 42
) -> tuple:
    """Train RandomForest model and return model + metrics."""
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_absolute_error, r2_score

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=12,
        min_samples_leaf=5,
        random_state=random_state,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    return model, {"mae_kw": mae, "r2": r2}, (X_test, y_test, y_pred)


CHUNK_CUSTOMERS = 100  # Process this many customers at a time to avoid OOM on large sources


def predict_ac_from_parquet(model, path: Path) -> pd.DataFrame:
    """Run predict_ac on parquet file in row-group batches to avoid OOM."""
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(path)
    parts = []
    for i in range(pf.num_row_groups):
        table = pf.read_row_group(i)
        df_chunk = table.to_pandas()
        part = predict_ac(model, df_chunk)
        parts.append(part)
    return pd.concat(parts, ignore_index=True)


def predict_ac(model, df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply trained model to estimate AC load for any rows with TOT + weather.
    Processes by source, and for large sources by customer chunks.
    """
    feature_cols = ["TOT", "temp", "glob_rad", "hour", "dow", "month"]
    parts = []
    for source_name, grp in df.groupby("source", sort=False):
        n_cust = grp["id_customer"].nunique()
        if n_cust > CHUNK_CUSTOMERS:
            # Chunk by id_customer to avoid OOM on large sources (e.g. RE)
            cust_ids = grp["id_customer"].unique()
            for i in range(0, len(cust_ids), CHUNK_CUSTOMERS):
                batch_ids = cust_ids[i : i + CHUNK_CUSTOMERS]
                sub = grp[grp["id_customer"].isin(batch_ids)]
                part = _predict_ac_single_group(model, sub, feature_cols)
                if part is not None:
                    parts.append(part)
        else:
            part = _predict_ac_single_group(model, grp, feature_cols)
            if part is not None:
                parts.append(part)
    if not parts:
        raise ValueError("Data must have TOT (total load) type.")
    return pd.concat(parts, ignore_index=True)


def _predict_ac_single_group(model, grp: pd.DataFrame, feature_cols: list) -> pd.DataFrame | None:
    """Pivot, build features, predict for one group."""
    pivoted = load_and_pivot(grp)
    if "TOT" not in pivoted.columns:
        return None
    pivoted = build_features(pivoted)
    weather = (
        grp[["source", "dt_utc", "temp", "glob_rad"]]
        .drop_duplicates(subset=["source", "dt_utc"])
    )
    pivoted = pivoted.merge(weather, on=["source", "dt_utc"], how="left")
    pivoted["glob_rad"] = pivoted["glob_rad"].fillna(0)
    pivoted["temp"] = pivoted["temp"].fillna(pivoted["temp"].median())
    X = pivoted[feature_cols]
    ac_pred = model.predict(X)
    tot_vals = pivoted["TOT"].values
    ac_pred = np.clip(ac_pred, 0, tot_vals)
    pivoted["ac_pred_kw"] = ac_pred
    return pivoted


def plot_extracted_ac(result: pd.DataFrame, output_path: Path) -> None:
    """
    Plot sample curves: total load vs predicted AC.
    Picks one customer per source and plots 7 days.
    """
    import matplotlib.pyplot as plt

    result = result.copy()
    result["dt_utc"] = pd.to_datetime(result["dt_utc"], utc=True)

    # Sample one customer per source
    sources = result["source"].unique()
    n_sources = min(3, len(sources))
    fig, axes = plt.subplots(n_sources, 1, figsize=(12, 3 * n_sources), sharex=True)
    if n_sources == 1:
        axes = [axes]

    for i, src in enumerate(sources[:n_sources]):
        subset = result[result["source"] == src]
        cust = subset["id_customer"].iloc[0]
        sub = subset[subset["id_customer"] == cust].sort_values("dt_utc")
        # Take 7 days
        sub = sub.iloc[: 4 * 24 * 7]  # 15min intervals, 7 days
        if sub.empty:
            continue
        ax = axes[i]
        ax.plot(sub["dt_utc"], sub["TOT"], label="Total (kW)", alpha=0.8)
        ax.plot(sub["dt_utc"], sub["ac_pred_kw"], label="AC predicted (kW)", alpha=0.8)
        ax.set_ylabel("kW")
        ax.set_title(f"{src} – id_customer={cust}")
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("UTC")
    fig.suptitle("AC Load Extraction: Total vs Predicted AC")
    plt.tight_layout()
    fig.savefig(output_path, dpi=100)
    plt.show()
    print(f"Plot saved to {output_path}")


def validate_and_plot_re(result: pd.DataFrame, output_path: Path) -> dict:
    """
    Run sanity checks on RE (or any source without ground truth) predictions
    and save validation plot.
    Returns summary dict.
    """
    import matplotlib.pyplot as plt

    result = result.copy()
    result["dt_utc"] = pd.to_datetime(result["dt_utc"], utc=True)
    result["hour"] = result["dt_utc"].dt.hour
    result["month"] = result["dt_utc"].dt.month

    summary = {}

    # 1. AC ≤ TOT
    ok = (result["ac_pred_kw"] <= result["TOT"]).all()
    summary["ac_le_tot"] = "PASS" if ok else "FAIL (some AC > TOT)"
    n_violate = (result["ac_pred_kw"] > result["TOT"]).sum()
    summary["ac_le_tot_violations"] = int(n_violate)

    # 2. Temp–AC correlation (positive expected)
    corr = result["ac_pred_kw"].corr(result["temp"])
    summary["temp_ac_corr"] = float(corr)

    # 3. Mean AC / Mean TOT ratio
    mean_ac = result["ac_pred_kw"].mean()
    mean_tot = result["TOT"].mean()
    summary["mean_ac_kw"] = float(mean_ac)
    summary["mean_tot_kw"] = float(mean_tot)
    summary["ac_share_pct"] = 100 * mean_ac / mean_tot if mean_tot > 0 else 0

    print("\n[RE/No-GT validation]")
    for k, v in summary.items():
        print(f"  {k}: {v}")

    # Plot: temp vs AC scatter, hourly profile, monthly profile
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    # Scatter: temp vs AC (sample)
    sample = result.sample(n=min(5000, len(result)), random_state=42)
    axes[0].scatter(sample["temp"], sample["ac_pred_kw"], alpha=0.3, s=5)
    axes[0].set_xlabel("Temperature (°C)")
    axes[0].set_ylabel("AC predicted (kW)")
    axes[0].set_title(f"Temp vs AC (corr={corr:.3f})")
    axes[0].grid(True, alpha=0.3)

    # Hourly profile
    hourly = result.groupby("hour")["ac_pred_kw"].mean()
    axes[1].bar(hourly.index, hourly.values)
    axes[1].set_xlabel("Hour (UTC)")
    axes[1].set_ylabel("Mean AC (kW)")
    axes[1].set_title("Hourly pattern")
    axes[1].grid(True, alpha=0.3)

    # Monthly profile
    monthly = result.groupby("month")["ac_pred_kw"].mean()
    axes[2].bar(monthly.index, monthly.values)
    axes[2].set_xlabel("Month")
    axes[2].set_ylabel("Mean AC (kW)")
    axes[2].set_title("Monthly pattern (seasonality)")
    axes[2].grid(True, alpha=0.3)

    fig.suptitle("RE AC Prediction – Sanity Validation")
    plt.tight_layout()
    fig.savefig(output_path, dpi=100)
    plt.close()
    print(f"Validation plot saved to {output_path}")

    return summary


def plot_validation(X_test, y_test, y_pred, output_path: Path) -> None:
    """Plot ground truth vs predicted AC (Dataport validation)."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    # Scatter
    axes[0].scatter(y_test, y_pred, alpha=0.3, s=5)
    max_val = max(y_test.max(), y_pred.max())
    axes[0].plot([0, max_val], [0, max_val], "r--", label="y=x")
    axes[0].set_xlabel("Ground truth AC (kW)")
    axes[0].set_ylabel("Predicted AC (kW)")
    axes[0].set_title("AC: Ground Truth vs Predicted")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    # Time series sample (first 500 points)
    n = min(500, len(y_test))
    axes[1].plot(range(n), y_test.values[:n], label="Ground truth", alpha=0.8)
    axes[1].plot(range(n), y_pred[:n], label="Predicted", alpha=0.8)
    axes[1].set_xlabel("Sample index")
    axes[1].set_ylabel("AC (kW)")
    axes[1].set_title("AC over time (sample)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    fig.suptitle("AC Load Model Validation (Dataport)")
    plt.tight_layout()
    fig.savefig(output_path, dpi=100)
    plt.show()
    print(f"Validation plot saved to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="AC load extraction: train and/or predict on RE data")
    parser.add_argument(
        "--re-data-dir",
        type=str,
        default=None,
        help=f"Path to RE parquet folder (e.g. {DEFAULT_RE_DATA_DIR}). "
        "If set, loads all parquets, fetches weather, and writes ac_load_extracted_RE.parquet",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Show and save plots: validation (GT vs pred) + extracted AC curves",
    )
    args = parser.parse_args()

    if not UNIFIED_PATH.exists():
        raise FileNotFoundError(f"Unified parquet not found: {UNIFIED_PATH}")

    df_dataport = load_dataport_only()
    print(f"Loaded {len(df_dataport):,} Dataport rows for training")

    X, y, train_df = prepare_training_data(df_dataport)
    print(f"Training samples (Dataport with AC): {len(X):,}")

    model, metrics, (X_test, y_test, y_pred) = train_and_evaluate(X, y)
    print(f"MAE (kW): {metrics['mae_kw']:.4f}")
    print(f"R²: {metrics['r2']:.4f}")

    # Feature importance
    feature_cols = list(X.columns)
    imp = pd.Series(model.feature_importances_, index=feature_cols).sort_values(
        ascending=False
    )
    print("\nFeature importance:")
    for name, val in imp.items():
        print(f"  {name}: {val:.3f}")

    # Optionally save model (requires joblib)
    try:
        import joblib
        model_path = MODEL_DIR / "ac_load_model.joblib"
        joblib.dump({"model": model, "feature_cols": feature_cols}, model_path)
        print(f"\nModel saved to {model_path}")
    except ImportError:
        print("\n(joblib not installed, skipping model save)")

    # Apply to full dataset and save AC predictions (in batches to avoid OOM)
    result = predict_ac_from_parquet(model, UNIFIED_PATH)
    ac_only = result[["source", "id_customer", "dt_utc", "ac_pred_kw"]].copy()
    ac_only = ac_only.rename(columns={"ac_pred_kw": "value_kw_mean"})
    ac_only["type"] = "AC"
    output_path = PROCESSED_DIR / "ac_load_extracted.parquet"
    ac_only.to_parquet(output_path, index=False)
    print(f"\nAC load predictions written to {output_path}")

    if args.plot:
        plot_validation(X_test, y_test, y_pred, PROCESSED_DIR / "ac_validation.png")
        plot_extracted_ac(result, PROCESSED_DIR / "ac_extracted_curves.png")

    # Optionally run on RE data
    if args.re_data_dir:
        from data.load_smart_meter import load_re_for_ac

        re_dir = Path(args.re_data_dir)
        if not re_dir.exists():
            fallback = Path(DEFAULT_RE_DATA_DIR)
            if fallback.exists():
                print(f"RE path not found: {re_dir}, using default: {fallback}")
                re_dir = fallback
            else:
                raise FileNotFoundError(
                    f"RE data dir not found: {re_dir}. "
                    f"Check --re-data-dir path (no space before .venv etc)."
                )
        print(f"\nLoading RE data from {re_dir}...")
        re_df = load_re_for_ac(re_dir, add_weather=True)
        print(f"Loaded {len(re_df):,} RE rows")
        re_result = predict_ac(model, re_df)
        re_ac = re_result[["source", "id_customer", "dt_utc", "ac_pred_kw"]].copy()
        re_ac = re_ac.rename(columns={"ac_pred_kw": "value_kw_mean"})
        re_ac["type"] = "AC"
        re_output = PROCESSED_DIR / "ac_load_extracted_RE.parquet"
        re_ac.to_parquet(re_output, index=False)
        print(f"RE AC load predictions written to {re_output}")
        if args.plot:
            plot_extracted_ac(re_result, PROCESSED_DIR / "ac_extracted_curves_RE.png")
        validate_and_plot_re(
            re_result, PROCESSED_DIR / "ac_re_validation.png"
        )

    return model, result


# %%
if __name__ == "__main__":
    main()
