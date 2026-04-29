from pathlib import Path
import pandas as pd

PATH = Path("data/processed_data/heapo_hp_long.parquet")

df = pd.read_parquet(PATH)

print("rows:", len(df))
print("cols:", df.columns.tolist())

print("\nunique customers:", df["ID customer"].nunique())
print("types:", df["type"].unique())

print("\ndt range:", df["dt_utc"].min(), "->", df["dt_utc"].max())

print("\nValue stats (kW):")
print(df["Value_KW_mean"].describe())

print("\nshare zero:", (df["Value_KW_mean"] == 0).mean())

# Check one customer sample's time step 
cid = df["ID customer"].iloc[0]
s = df[df["ID customer"] == cid].sort_values("dt_utc").head(40)
diffs = s["dt_utc"].diff().dropna().value_counts()
print("\nTime step counts (sample customer):")
print(diffs.head(5).to_string())