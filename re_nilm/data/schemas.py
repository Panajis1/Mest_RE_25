"""Column-level documentation and type definitions for all DataFrames used in the pipeline."""

from typing import TypedDict


class SmartMeterRow(TypedDict):
    ID: str
    DT_UTC: object       # pd.Timestamp (tz-naive UTC)
    CONSO_KWH: float
    PROD_KWH: float


class WeatherRow(TypedDict):
    dt_utc: object       # pd.Timestamp (tz-naive UTC)
    t_2m_C: float
    global_rad_W: float


class DailyFeaturesRow(TypedDict):
    customer_id: str
    date: object         # datetime.date
    corr_temp_all: float
    corr_temp_cold: float
    corr_temp_hot: float
    season_balance: float
    thermal_balance: float
    coeff_var: float
    acf_1h: float
    acf_24h: float
    G_daily: float
    G_midday: float


class DetectionResultRow(TypedDict):
    customer_id: str
    has_pv: bool
    prob_pv: float
    has_ac: bool
    prob_ac: float
    has_hp: bool
    prob_hp: float
    hp_type: str         # "no_hp" | "winter_hp" | "summer_hp"
    has_battery: bool
    prob_battery: float


class CapacityResultRow(TypedDict):
    customer_id: str
    pv_capacity_kwp: float
    pv_ci_lower: float
    pv_ci_upper: float
    sc_share: float
    sc_ci_lower: float
    sc_ci_upper: float
    battery_capacity_kwh: float
    batt_ci_lower: float
    batt_ci_upper: float
