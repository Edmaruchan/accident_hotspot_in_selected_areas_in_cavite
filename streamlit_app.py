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
st.subheader("Welcome to the Road Accident Dashboard")
st.write("Select a municipality to view clustering results and maps.")
page = st.radio("Select Area", ["Overview", "Alfonso", "GMA", "Carmona"])

# Pages
if page == "Overview":
  st.write("")
  st.subheader("Road Accident Hotspot Maps")
  col1, col2 = st.columns(2)
  with col1:
     st.image("data/qgis_maps/all.png", caption="Map")
  with col2:
     st.image("data/qgis_maps/alfonso.png", caption="Alfonso Map")
     st.image("data/qgis_maps/carmona.png", caption="Carmona Map")
     st.image("data/qgis_maps/gma.png", caption="GMA Map")
        
  

elif page == "Alfonso":
    st.subheader("Alfonso Analysis")
    st.image("data/qgis_maps/alfonso.png", caption="Hotspots in Alfonso")
    # Show stats, graphs, etc.
elif page == "GMA":
    st.subheader("GMA Analysis")
    st.image("data/qgis_maps/gma.png", caption="Hotspots in GMA")
    st.image("data/qgis_maps/GMA/gma_heatmap.png", caption="Accident Heatmap in GMA")
    st.image("data/qgis_maps/GMA/gma_month.png", caption="W.I.P")
    st.image("data/qgis_maps/GMA/gma_year_1.png", caption="W.I.P")
elif page == "Carmona":
    st.subheader("Carmona Analysis")
    st.image("data/qgis_maps/carmona.png", caption="Hotspots in Carmona")


# Sidebar – Controls
st.sidebar.title("Display Options")
show_clustering = st.sidebar.checkbox("Enable Interactive Clustering")

if show_clustering:
    method = st.sidebar.radio("Select Clustering Method", ["DBSCAN", "KMeans"])
    if method == "KMeans":
        k = st.sidebar.slider("Number of Clusters", min_value=1, max_value=10, value=3)
    
    if st.button("Run Clustering"):
        # Load data and define coords
        df = pd.read_csv("data/ALL.csv")
        coords = df[['Latitude', 'Longitude']].values

        if method == "DBSCAN":
            st.subheader("DBSCAN Clustering")
            coords_rad = np.radians(coords)
            eps_rad = 0.3 / 6371.0  # 300 meters
            db = DBSCAN(eps=eps_rad, min_samples=3, metric='haversine')
            df['cluster'] = db.fit_predict(coords_rad)

            fig, ax = plt.subplots(figsize=(8, 6))
            scatter = ax.scatter(df['Longitude'], df['Latitude'], c=df['cluster'], cmap='tab20', s=10)
            ax.set_title("DBSCAN Clustering of Road Accidents in 3 selected municipalities")
            ax.set_xlabel("Longitude")
            ax.set_ylabel("Latitude")
            ax.grid(True)
            st.pyplot(fig)

        elif method == "KMeans":
            st.subheader("KMeans Clustering")
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
