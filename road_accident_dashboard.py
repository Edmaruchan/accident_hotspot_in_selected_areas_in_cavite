import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

import streamlit as st
from PIL import Image

st.title("Road Accident Analysis in Cavite")


# Sidebar or top menu

page = st.radio("Select Area", ["Overview", "Alfonso", "GMA", "Carmona"])

# Pages
if page == "Overview":
  st.subheader("Welcome to the Road Accident Dashboard")
  st.write("Select a municipality to view clustering results and maps.")

elif page == "Alfonso":
    st.subheader("Alfonso Analysis")
    st.image("qgis_maps/alfonso_map.png", caption="Hotspots in Alfonso")
    # Show stats, graphs, etc.
elif page == "GMA":
    st.subheader("GMA Analysis")
    st.image("qgis_maps/gma_map.png", caption="Hotspots in GMA")
elif page == "Carmona":
    st.subheader("Carmona Analysis")
    st.image("qgis_maps/carmona_map.png", caption="Hotspots in Carmona")

# Sidebar – Controls
st.sidebar.title("Clustering Controls")
method = st.sidebar.radio("Select Clustering Method", ["DBSCAN", "KMeans"])

    

df = pd.read_csv("data/ALL.csv")

# Show preview
#st.write("Dataset Preview", df.head())

# Extract coordinates
coords = df[['Latitude', 'Longitude']].values

# Button to run clustering
if st.button("Run Clustering"):

    if method == "DBSCAN":
        st.subheader("DBSCAN Clustering")
        coords_rad = np.radians(coords)
        eps_rad = 0.3 / 6371.0  # 300 meters
        db = DBSCAN(eps=eps_rad, min_samples=3, metric='haversine')
        df['cluster'] = db.fit_predict(coords_rad)
        
      #  st.write("Cluster Counts:", df['cluster'].value_counts())

        fig, ax = plt.subplots(figsize=(8, 6))
        scatter = ax.scatter(df['Longitude'], df['Latitude'], c=df['cluster'], cmap='tab20', s=10)
        ax.set_title("DBSCAN Clustering of Road Accidents")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.grid(True)
        st.pyplot(fig)

    elif method == "KMeans":
        st.subheader("KMeans Clustering")
        k = st.slider("Number of Clusters", min_value=1, max_value=10, value=1)
        kmeans = KMeans(n_clusters=k, random_state=42)
        df['cluster'] = kmeans.fit_predict(coords)
        centers = kmeans.cluster_centers_

        st.write("Cluster Counts:", df['cluster'].value_counts())

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(df['Longitude'], df['Latitude'], c=df['cluster'], cmap='tab10', s=10)
        ax.scatter(centers[:, 1], centers[:, 0], c='black', marker='x', s=100, label='Centroids')
        ax.set_title(f"K-Means Clustering (k={k}) with Centroids")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)

# Main – Cluster Plot
st.title("Spatio-Temporal Road Accident Clustering")
df = pd.read_csv("data/ALL.csv")

# Show plot only if button clicked
if st.button("Run Clustering"):
    

# Right Panel – QGIS Map Images
 st.subheader("QGIS Road Accident Hotspot Maps")

col1, col2 = st.columns(2)
with col1:
    st.image("qgis_maps/gma_map.png", caption="GMA")
    st.image("qgis_maps/carmona_map.png", caption="Carmona")
with col2:
    st.image("qgis_maps/alfonso_map.png", caption="Alfonso")