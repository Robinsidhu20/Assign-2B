import streamlit as st
import pandas as pd
import folium
from folium import PolyLine, Marker
from streamlit_folium import st_folium
import networkx as nx
from io import StringIO

# === Load processed data with geo ===
data_path = "C:/Swinburne/Intro to AI/2B/Processed Data/traffic_with_geo.csv"  
df = pd.read_csv(data_path)

# Drop rows without geo info
df = df.dropna(subset=["Geo_Latitude", "Geo_Longitude"])

# Get unique sites with lat/lon
site_info = df.drop_duplicates(subset="Site_ID")[[
    "Site_ID", "Location", "Geo_Latitude", "Geo_Longitude"
]].reset_index(drop=True)

# === Sidebar controls ===
st.sidebar.title("SCATS Path Finder")
site_options = site_info["Site_ID"].tolist()

start_site = st.sidebar.selectbox("Select Start Site", site_options)
end_site = st.sidebar.selectbox("Select End Site", site_options, index=1)
model_choice = st.sidebar.selectbox("Select Prediction Model", [
    "Dummy Travel Time", "LSTM Model", "Linear Regression", "Rule-Based"
])

show_path = st.sidebar.button("Calculate Shortest Path")

# === Create map ===
center_lat = site_info["Geo_Latitude"].astype(float).mean()
center_lon = site_info["Geo_Longitude"].astype(float).mean()

m = folium.Map(location=[center_lat, center_lon], zoom_start=12)

# Add all site markers
for _, row in site_info.iterrows():
    Marker(
        location=[row["Geo_Latitude"], row["Geo_Longitude"]],
        popup=f"{row['Site_ID']} - {row['Location']}",
        icon=folium.Icon(color='blue', icon='info-sign')
    ).add_to(m)

# === Dummy Graph for Path ===
# In real use: Load predicted travel times as weights
G = nx.Graph()
for _, row in site_info.iterrows():
    G.add_node(row["Site_ID"], pos=(row["Geo_Latitude"], row["Geo_Longitude"]))

# Connect each site to 5 nearest neighbors (just for demo)
from sklearn.neighbors import NearestNeighbors
import numpy as np

coords = site_info[["Geo_Latitude", "Geo_Longitude"]].astype(float).to_numpy()
nbrs = NearestNeighbors(n_neighbors=6, algorithm='ball_tree').fit(coords)
distances, indices = nbrs.kneighbors(coords)

for idx, neighbors in enumerate(indices):
    src_id = site_info.loc[idx, "Site_ID"]
    for n in neighbors[1:]:
        dst_id = site_info.loc[n, "Site_ID"]
        # Add dummy weight (Euclidean distance)
        lat1, lon1 = coords[idx]
        lat2, lon2 = coords[n]
        dist = np.sqrt((lat1 - lat2) ** 2 + (lon1 - lon2) ** 2)
        G.add_edge(src_id, dst_id, weight=dist)

# === Calculate and Draw Path ===
if show_path:
    try:
        path = nx.shortest_path(G, source=start_site, target=end_site, weight="weight")
        path_coords = [
            (
                float(site_info[site_info["Site_ID"] == sid]["Geo_Latitude"].values[0]),
                float(site_info[site_info["Site_ID"] == sid]["Geo_Longitude"].values[0])
            ) for sid in path
        ]

        PolyLine(path_coords, color="red", weight=5, tooltip="Shortest Path").add_to(m)
        st.success(f"Shortest path found using: {model_choice}")
        st.text(" -> ".join(path))
    except Exception as e:
        st.error(f"Error calculating path: {e}")

# === Display the map ===
st_data = st_folium(m, width=700, height=500)
