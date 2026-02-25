from pathlib import Path
import pandas as pd

# input folder containing raw CSV files
INPUT_DIR = Path("data/raw_data/heapo")

# output file for processed data
OUTPUT_FILE = Path("data/processed_data/heapo_hp_long.parquet")

SOURCE = "Heapo"
TYPE_VALUE = "HP"


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

        # Timestamp → datetime (UTC)
        df["Timestamp"] = pd.to_datetime(df["Timestamp"], utc=True)

        # 15 min kWh → average kW
        df["HP_kW"] = pd.to_numeric(df["kWh_received_HeatPump"], errors="coerce") * 4

        df_out = pd.DataFrame({
            "Source": SOURCE,
            "ID customer": df["Household_ID"],
            "type": TYPE_VALUE,
            "Value_KW_mean": df["HP_kW"],
            "dt_utc": df["Timestamp"],
        })

        dfs.append(df_out)

    all_data = pd.concat(dfs, ignore_index=True).dropna()

    # drop overlapping timestamps
    all_data = (
        all_data.groupby(
            ["Source", "ID customer", "type", "dt_utc"],
            as_index=False
        )["Value_KW_mean"]
        .mean()
        .sort_values(["ID customer", "dt_utc"])
    )

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    all_data.to_parquet(OUTPUT_FILE, index=False)

    print("✅ Saved:", OUTPUT_FILE)
    print(all_data.head())


if __name__ == "__main__":
    main()