import pandas as pd
import matplotlib.pyplot as plt


# Define the path where the file was saved
file_path = "data/preprocessed_data/caltech_15min_kw.parquet"

try:
    # Load the Parquet file
    df_verify = pd.read_parquet(file_path)

    # Display the first 10 rows
    print("--- First 10 Rows of Processed Data ---")
    print(df_verify.head(10))

    # Verify data types and non-null counts
    print("\n--- Data Information ---")
    print(df_verify.info())

    # Check for UTC timezone specifically
    print(f"\nTimezone detected: {df_verify['dt_utc'].dt.tz}")

except FileNotFoundError:
    print(f"Error: The file at {file_path} was not found. Please run the fetcher script first.")

# --- Additional Check: Plotting the first few sessions ---
df = pd.read_parquet(file_path)

# 2. Select the first few unique customers to display
unique_customers = df['ID customer'].unique()[:5]
subset_df = df[df['ID customer'].isin(unique_customers)]

# 3. Create the plot
plt.figure(figsize=(12, 6))

for customer in unique_customers:
    # Filter and sort by time for a continuous line
    cust_data = subset_df[subset_df['ID customer'] == customer].sort_values('dt_utc')
    plt.plot(cust_data['dt_utc'], cust_data['Value_KW_mean'], marker='o', label=f'ID: {customer}')

# Formatting
plt.title('EV Charging Profiles - First Few Sessions (15-min mean kW)', fontsize=14)
plt.xlabel('Timestamp (UTC)', fontsize=12)
plt.ylabel('Power (kW)', fontsize=12)
plt.legend(title='Customer ID')
plt.grid(True, linestyle='--', alpha=0.6)
plt.xticks(rotation=45)
plt.tight_layout()

# 4. Display or save
plt.show()