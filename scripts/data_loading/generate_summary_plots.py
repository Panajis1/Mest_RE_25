import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

def generate_plots():
    # Set seaborn style similar to the attached figures
    sns.set_theme(style="whitegrid")
    
    # Custom color palettes imitating figures
    colors_pie = ['#fbb4ae', '#e3d5b8', '#ded2e6', '#ccebc5', '#decbe4', '#fed9a6', '#ffffcc', '#e5d8bd', '#fddaec']

    input_file = '/Users/alan/Desktop/ETH/cs_re/Mest_RE_25/data/processed_data/all_sources_load_with_weather.parquet'
    output_dir = '/Users/alan/Desktop/ETH/cs_re/Mest_RE_25/analysis_plots'
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("Loading data...")
    # Read relevant columns to save memory
    df = pd.read_parquet(input_file, columns=['type', 'source', 'dt_utc', 'dt_local', 'value_kw_mean', 'id_customer'])
    
    print("Filtering out EV 0-values for 'average spike' calculation...")
    # For EV types, we only look at non-zero data so the average doesn't smooth out the spikes
    ev_mask = df['type'].astype(str).str.contains('EV', na=False)
    zero_mask = (df['value_kw_mean'] > -1e-6) & (df['value_kw_mean'] < 1e-6)
    df = df[~(ev_mask & zero_mask)]
    
    print("Data loaded. Extracting time features...")
    # Fix missing dt_local for sources like 'pv estonia'
    if 'dt_utc' in df.columns:
        missing_local = df['dt_local'].isna()
        if missing_local.any():
            print(f"Fixing {missing_local.sum()} missing dt_local values using dt_utc...")
            if df['dt_utc'].dt.tz is not None:
                # Assuming missing local times (like Estonia) are UTC+2/3, we can just use Europe/Tallinn for them
                # or simpler, if source is 'pv estonia' apply Europe/Tallinn
                estonia_mask = missing_local & (df['source'] == 'pv estonia')
                if estonia_mask.any():
                    df.loc[estonia_mask, 'dt_local'] = df.loc[estonia_mask, 'dt_utc'].dt.tz_convert('Europe/Tallinn').dt.tz_localize(None)
                
                # SCIENTIFIC DATA is from Germany, so use Europe/Berlin
                scientific_mask = missing_local & (df['source'] == 'SCIENTIFIC DATA')
                if scientific_mask.any():
                    df.loc[scientific_mask, 'dt_local'] = df.loc[scientific_mask, 'dt_utc'].dt.tz_convert('Europe/Berlin').dt.tz_localize(None)
                
                # For any other remaining missing, simply drop tz
                still_missing = df['dt_local'].isna()
                if still_missing.any():
                    df.loc[still_missing, 'dt_local'] = df.loc[still_missing, 'dt_utc'].dt.tz_localize(None)
            else:
                df['dt_local'] = df['dt_local'].fillna(df['dt_utc'])
    
    # Fix time shift for Dataport (it is 5 hours ahead of local time, essentially UTC)
    dataport_mask = df['source'] == 'dataport'
    if dataport_mask.any():
        df.loc[dataport_mask, 'dt_local'] = df.loc[dataport_mask, 'dt_local'] - pd.Timedelta(hours=5)
    
    # Use dt_local to ensure no timezone shift issues, specifically requested by user.
    df['hour'] = df['dt_local'].dt.hour
    df['month'] = df['dt_local'].dt.month
    
    print("Normalizing PV data to absolute values...")
    # Normalize PV data so it is consistently positive across all sources
    pv_mask = df['type'] == 'PV'
    df.loc[pv_mask, 'value_kw_mean'] = df.loc[pv_mask, 'value_kw_mean'].abs()
    
    # ---------------------------------------------------------
    # 1. Sample amount pie charts (Customers per source / type)
    # ---------------------------------------------------------
    print("Generating pie charts for sample amounts...")
    customer_counts = df.groupby(['source', 'type'])['id_customer'].nunique().reset_index()
    
    # Customers by Source
    plt.figure(figsize=(10, 8))
    source_counts = customer_counts.groupby('source')['id_customer'].sum()
    plt.pie(source_counts, labels=source_counts.index, autopct='%1.1f%%', colors=colors_pie, startangle=90)
    plt.title('Sample Amount (Unique Customers) by Source')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'pie_customers_by_source.png'), dpi=300)
    plt.close()
    
    # Customers by Appliance Type
    plt.figure(figsize=(10, 8))
    type_counts = customer_counts.groupby('type')['id_customer'].sum()
    textprops = {'fontsize': 12}
    plt.pie(type_counts, labels=type_counts.index, autopct='%1.1f%%', colors=colors_pie, startangle=90, textprops=textprops)
    plt.title('Sample Amount (Unique Customers) by Appliance Type')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'pie_customers_by_type.png'), dpi=300)
    plt.close()

    # ---------------------------------------------------------
    # 2. Average Size bar chart (Mean value_kw_mean per type/source)
    # ---------------------------------------------------------
    print("Generating average size bar charts...")
    avg_size = df.groupby(['source', 'type'])['value_kw_mean'].mean().reset_index()
    
    plt.figure(figsize=(12, 6))
    sns.barplot(data=avg_size, x='source', y='value_kw_mean', hue='type', palette='deep')
    plt.title('Average Size (kW) by Source and Appliance Type')
    plt.ylabel('Average kW')
    plt.xlabel('Source')
    plt.xticks(rotation=45)
    plt.legend(title='Appliance Type', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'bar_avg_size_source_type.png'), dpi=300)
    plt.close()
    
    # ---------------------------------------------------------
    # 3. Average Daily Figures (Per Type and Source)
    # ---------------------------------------------------------
    print("Generating average daily profiles...")
    hourly_avg = df.groupby(['source', 'type', 'hour'])['value_kw_mean'].mean().reset_index()
    
    unique_types = hourly_avg['type'].unique()
    for t in unique_types:
        data_t = hourly_avg[hourly_avg['type'] == t]
        
        plt.figure(figsize=(12, 6))
        for i, source in enumerate(data_t['source'].unique()):
            data_s = data_t[data_t['source'] == source]
            color = sns.color_palette("deep")[i % 10]
            
            plt.plot(data_s['hour'], data_s['value_kw_mean'], label=f'{source}', color=color)
            plt.fill_between(data_s['hour'], data_s['value_kw_mean'], alpha=0.2, color=color)
            
        plt.title(f'Average Daily Profile - {t}')
        plt.xlabel('Local Hour of Day')
        plt.ylabel('Avg Power (kW)' if 'EV' not in t else 'Avg Charging Power (kW)')
        plt.xticks(range(0, 25, 2))
        plt.legend(title='Source')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'daily_profile_{t}.png'), dpi=300)
        plt.close()

    # ---------------------------------------------------------
    # 4. Yearly Figures (Average per Month per Type and Source)
    # ---------------------------------------------------------
    print("Generating yearly profiles...")
    monthly_avg = df.groupby(['source', 'type', 'month'])['value_kw_mean'].mean().reset_index()
    
    for t in unique_types:
        data_t = monthly_avg[monthly_avg['type'] == t]
        
        plt.figure(figsize=(12, 6))
        for i, source in enumerate(data_t['source'].unique()):
            data_s = data_t[data_t['source'] == source]
            color = sns.color_palette("deep")[i % 10]
            
            plt.plot(data_s['month'], data_s['value_kw_mean'], label=f'{source}', color=color, marker='o')
            plt.fill_between(data_s['month'], data_s['value_kw_mean'], alpha=0.2, color=color)
            
        plt.title(f'Yearly Profile (Monthly Avg) - {t}')
        plt.xlabel('Month')
        plt.ylabel('Avg Power (kW)' if 'EV' not in t else 'Avg Charging Power (kW)')
        plt.xticks(range(1, 13), ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
        plt.legend(title='Source')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'yearly_profile_{t}.png'), dpi=300)
        plt.close()
        
    print(f"All plots have been successfully generated and saved to: {output_dir}")

if __name__ == "__main__":
    generate_plots()
