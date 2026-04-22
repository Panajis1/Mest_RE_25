# Data Summarization Visualization Report

## Project Overview
This document summarizes the `all_sources_load_with_weather.parquet` dataset and compares it against historical figures from **Romande Energy**.

### Romande Energy Data Overview
* **Total Unique Customers:** 2,916
* **Prosumers (with PV):** 2,893 (99.2%)
* **Consumers Only:** 23 (0.8%)
* **Total Energy Consumed:** 26.54 GWh
* **Total Energy Produced:** 182.68 GWh
* **Ratio:** ~6.9x -> Much more generation than consumption.

*(Here are the reference Romande Energy historical plots for comparison)*

**2-Week Profile:**
![Romande Energy: 2-Week Profile (Customer 1)](data/out/top5_c1_timeseries.png)

**Average Daily Profile:**
![Romande Energy: Average Daily Profile (Customer 1)](data/out/top5_c2_daily_profile.png)

**Customer Base by Annual Consumption Size:**
![Romande Energy: Customer Base by Annual Consumption Size](data/out/size_distribution_pie.png)

---

## Dataset Summaries

### Sample Amounts
The following charts display the distribution of unique customers by source and by appliance type.

![Customers by Source](analysis_plots/pie_customers_by_source.png)

![Customers by Appliance Type](analysis_plots/pie_customers_by_type.png)

### Average Size
This grouping compares the average load size (kW) across sources and types.

![Average Size by Source and Type](analysis_plots/bar_avg_size_source_type.png)

### Average Daily Profiles
Average power usage distributed over a 24-hour period, comparing different sources per appliance type.

````carousel
![PV Daily Profile](analysis_plots/daily_profile_PV.png)
<!-- slide -->
![HP Daily Profile](analysis_plots/daily_profile_HP.png)
<!-- slide -->
![AC Daily Profile](analysis_plots/daily_profile_AC.png)
<!-- slide -->
![EV Daily Profile](analysis_plots/daily_profile_EV.png)
<!-- slide -->
![TOT Daily Profile](analysis_plots/daily_profile_TOT.png)
````

### Yearly Profiles
Average power distribution over the months of the year, comparing different sources per appliance type.

````carousel
![PV Yearly Profile](analysis_plots/yearly_profile_PV.png)
<!-- slide -->
![HP Yearly Profile](analysis_plots/yearly_profile_HP.png)
<!-- slide -->
![AC Yearly Profile](analysis_plots/yearly_profile_AC.png)
<!-- slide -->
![EV Yearly Profile](analysis_plots/yearly_profile_EV.png)
<!-- slide -->
![TOT Yearly Profile](analysis_plots/yearly_profile_TOT.png)
````
