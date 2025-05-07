# SCATS Traffic Data – Processed Dataset (October 2006)

This dataset contains cleaned and time-aligned traffic volume records from VicRoads' SCATS system. It has been processed for both machine learning and exploratory data analysis (EDA) tasks.

---

## Files Generated

- traffic_model_ready.csv – Cleaned dataset for ML training (includes timestamp, volume, and location info)
- traffic_model_ready.pkl – Same as above, in faster binary format
- traffic_with_geo.csv – Extended dataset with geo-coordinates and descriptive fields
- traffic_with_geo.pkl – Same as above, in binary format

---

## Columns in traffic_model_ready.csv

- Site_ID: SCATS site number
- Location: Text label of the road/intersection
- Latitude: From SCATS dataset
- Longitude: From SCATS dataset
- Timestamp: Combined date and 15-minute time slot
- Volume: Vehicle count for that 15-minute window

---

## Sites Missing Geo Metadata

The following Site_IDs were not found in the geolocation mapping file and have missing Geo_Latitude, Geo_Longitude, and SITE_DESC:

- 2000
- 2846
- 3804

These are still included in the output for completeness.

---

## How to Use

For modeling:
- Use traffic_model_ready.pkl or .csv
- You can group by Site_ID or resample over time using Timestamp
- Suitable for time series models like LSTM, GRU, ARIMA, or Prophet

For EDA and visualization:
- Use traffic_with_geo.csv
- Includes descriptive columns and GPS for mapping and filtering

---

## Processing Notes

- Wide format V00–V95 columns were melted into long format
- Timestamps were constructed from Date + 15-minute slot
- Missing values and invalid rows were removed
- Site_IDs were padded with leading zeros to ensure proper merging
