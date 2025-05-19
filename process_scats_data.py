import pandas as pd
from datetime import timedelta

# === File paths ===
TRAFFIC_DATA = "Scats Data October 2006/Scats Data October 2006.csv" #Change as per your own specified location
GEO_DATA = "Traffic_Count_Locations_with_LONG_LAT.csv" #Change as per your own specified location

# === Load SCATS traffic data ===
df = pd.read_csv(TRAFFIC_DATA)

# Fix column headers
df.columns = df.iloc[0]  # Set actual headers
df = df[1:]              # Drop the duplicated header row
df.columns = [str(c).strip() for c in df.columns]

# Create time mapping V00–V95 → 15 min increments
time_map = {f"V{str(i).zfill(2)}": timedelta(minutes=15 * i) for i in range(96)}

# Keep only necessary columns
meta_cols = ['SCATS Number', 'Location', 'NB_LATITUDE', 'NB_LONGITUDE', 'Date']
value_cols = list(time_map.keys())
df = df[meta_cols + value_cols].copy()

# Melt to long format
df_long = df.melt(
    id_vars=meta_cols,
    value_vars=value_cols,
    var_name="TimeCode",
    value_name="Volume"
)

# Build full timestamp
df_long["Date"] = pd.to_datetime(df_long["Date"], dayfirst=True, errors="coerce")
df_long["TimeDelta"] = df_long["TimeCode"].map(time_map)
df_long["Timestamp"] = df_long["Date"] + df_long["TimeDelta"]

# Clean and rename columns
df_long = df_long.rename(columns={
    'SCATS Number': 'Site_ID',
    'NB_LATITUDE': 'Latitude',
    'NB_LONGITUDE': 'Longitude'
})
df_long = df_long[['Site_ID', 'Location', 'Latitude', 'Longitude', 'Timestamp', 'Volume']]
df_long = df_long.dropna(subset=["Timestamp", "Volume"])
df_long["Volume"] = pd.to_numeric(df_long["Volume"], errors='coerce')
df_long = df_long.dropna(subset=["Volume"])

# Merge with geolocation data
geo_df = pd.read_csv(GEO_DATA)
geo_df = geo_df.rename(columns={
    "TFM_ID": "Site_ID",
    "X": "Geo_Longitude",
    "Y": "Geo_Latitude"
})

# Normalize Site_IDs with padding
geo_df["Site_ID"] = geo_df["Site_ID"].astype(str).str.lstrip('0')
df_long["Site_ID"] = df_long["Site_ID"].astype(str).str.lstrip('0')

# Merge
df_long = df_long.merge(
    geo_df[["Site_ID", "Geo_Latitude", "Geo_Longitude", "SITE_DESC"]],
    on="Site_ID", how="left"
)

# For Machine Learning Team (model-ready)
df_model = df_long[["Site_ID", "Location", "Latitude", "Longitude", "Timestamp", "Volume"]]
df_model.to_csv("Processed Data/traffic_model_ready.csv", index=False) #Change as per your own specified location
df_model.to_pickle("Processed Data/traffic_model_ready.pkl") #Change as per your own specified location
print("Saved: traffic_model_ready.csv and .pkl (model-ready dataset)")

# For Visualization/EDA Team (with geo info)
df_geo = df_long.copy()
df_geo.to_csv("Processed Data/traffic_with_geo.csv", index=False) #Change as per your own specified location
df_geo.to_pickle("Processed Data/traffic_with_geo.pkl") #Change as per your own specified location
print("Saved: traffic_with_geo.csv and .pkl (with location metadata)")