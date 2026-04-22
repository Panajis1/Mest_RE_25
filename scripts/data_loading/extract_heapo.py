from pathlib import Path
import pandas as pd

# input folder containing raw CSV files
INPUT_DIR = Path("data/raw_data/heapo")

# output file for processed data
OUTPUT_FILE = Path("data/processed_data/heapo_hp_long.parquet")

SOURCE = "Heapo"
TYPE_VALUE = "HP"

# Zurich timezone (handles DST automatically)
LOCAL_TZ = "Europe/Zurich"


def main():
    files = sorted(INPUT_DIR.glob("*.csv"))

    if not files:
        raise ValueError(f"No CSV files found in {INPUT_DIR.resolve()}")

    print(f"Found {len(files)} CSV files")

    dfs = []

    for f in files:
        print(f"Loading {f.name}")

        df = pd.read_csv(f, sep=";")

        # check required columns
        required_cols = {"Household_ID", "Timestamp", "kWh_received_HeatPump"}
        missing = required_cols - set(df.columns)
        if missing:
            raise KeyError(f"{f.name} missing columns: {missing}")

        # Timestamp -> datetime (UTC). Your Timestamp includes +00:00 already; utc=True enforces it safely.
        dt_utc = pd.to_datetime(df["Timestamp"], utc=True, errors="coerce")
        if dt_utc.isna().all():
            raise ValueError(f"{f.name}: failed to parse any Timestamp values")

        # Convert UTC to local Zurich time (keeps tz info, handles DST)
        dt_local = dt_utc.dt.tz_convert(LOCAL_TZ)

        # 15 min kWh -> average kW
        hp_kw = pd.to_numeric(df["kWh_received_HeatPump"], errors="coerce") * 4.0

        df_out = pd.DataFrame({
            "Source": SOURCE,
            "ID customer": df["Household_ID"],
            "type": TYPE_VALUE,
            "Value_KW_mean": hp_kw,
            "dt_utc": dt_utc,
            "dt_local": dt_local,
        })

        dfs.append(df_out)

    all_data = pd.concat(dfs, ignore_index=True)

    # Drop rows with missing essentials
    all_data = all_data.dropna(subset=["ID customer", "Value_KW_mean", "dt_utc", "dt_local"])

    # Drop overlapping timestamps (per customer)
    all_data = (
        all_data.groupby(["Source", "ID customer", "type", "dt_utc"], as_index=False)
        .agg(
            Value_KW_mean=("Value_KW_mean", "mean"),
            dt_local=("dt_local", "first"),  # unique mapping from dt_utc -> dt_local, so first is fine
        )
        .sort_values(["ID customer", "dt_utc"])
    )

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    all_data.to_parquet(OUTPUT_FILE, index=False)

    print("✅ Saved:", OUTPUT_FILE)
    print(all_data.head())


if __name__ == "__main__":
    main()