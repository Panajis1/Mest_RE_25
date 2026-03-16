# Romande Energie – Portfolio Summary

Number of customers in portfolio (after filters in `load_re_data`): **2916**.

## Yearly production and consumption

- **Yearly production (kWh)**: mean ≈ 62648.2, median ≈ 4802.0, 10–90% range ≈ [1242.6, 11304.3].
- **Yearly consumption (kWh)**: mean ≈ 9101.9, median ≈ 6393.6, 10–90% range ≈ [1772.4, 16942.9].

Histogram plots are saved under `analysis_plots` (yearly production, yearly consumption).

## Production-to-consumption ratio

- **Ratio distribution** (yearly production / yearly consumption, excluding customers with ~zero consumption): median ≈ 0.752, 10–90% range ≈ [0.148, 2.954].
- Customers with zero or extremely small yearly consumption are excluded from the ratio distribution but still counted in the total customer population.

The ratio histogram is saved as `production_to_consumption_ratio.png` in `analysis_plots`.

## Metadata overview

A metadata file was expected at `data/re_data/ETHZ/metadata`, but it could not be loaded in a recognised format.
