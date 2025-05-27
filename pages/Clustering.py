import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import plotly.express as px
import streamlit as st
from PIL import Image
from sklearn.neighbors import NearestNeighbors
import pyproj
from kneed import KneeLocator
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
import matplotlib.patches as mpatches
import folium
from streamlit_folium import st_folium
from sklearn import metrics
from sklearn.metrics import silhouette_score





st.title("Clustering")

page = st.radio("Select Area", ["Alfonso", "GMA", "Carmona"])


########################### ALFONSO ####################
if page == "Alfonso":

    st.write("Clustering in Alfonso")
    
    tab = st.radio("Select Clustering", ["K-Means", "DBSCAN"], key="alfonso")

    if tab == "K-Means":

        st.subheader("K-Means Clustering")
            
        col1, col2 = st.columns(2)
        
        with col1:

############# ELBOW METHOD ######################

            df = pd.read_csv("data/Alfonso/ALFONSO 2020 - 2024.csv")

            # Drop rows with missing lat/lon
            df = df.dropna(subset=['Latitude', 'Longitude'])

            # Use only coordinates for clustering
            coords = df[['Latitude', 'Longitude']]

            # Elbow method: compute inertia for a range of k
            inertias = []
            k_range = range(1, 11)

            for k in k_range:
                kmeans = KMeans(n_clusters=k, random_state=0, n_init='auto')
                kmeans.fit(coords)
                inertias.append(kmeans.inertia_)

            # Plot the Elbow chart
            fig_elbow_alfonso = go.Figure()
            fig_elbow_alfonso .add_trace(go.Scatter(x=list(k_range), y=inertias, mode='lines+markers'))
            fig_elbow_alfonso .update_layout(title="Elbow Method for Optimal k",
                                xaxis_title="Number of Clusters (k)",
                                yaxis_title="Inertia (Within-cluster sum of squares)")

            st.plotly_chart(fig_elbow_alfonso )


        with col2:
        ############### Automation of the elbow method ####################

            # Load and clean data
            df = pd.read_csv("data/Alfonso/ALFONSO 2020 - 2024.csv")
            df = df.dropna(subset=['Latitude', 'Longitude'])
            coords = df[['Latitude', 'Longitude']]

            # Compute inertia for k = 1 to 10
            inertias = []
            k_range = range(1, 11)
            for k in k_range:
                kmeans = KMeans(n_clusters=k, random_state=0, n_init='auto')
                kmeans.fit(coords)
                inertias.append(kmeans.inertia_)

        # Use kneed to detect the elbow
            knee = KneeLocator(k_range, inertias, curve='convex', direction='decreasing')
            optimal_k = knee.elbow

        # Show Elbow Plot
            fig_elbow_alfonso  = go.Figure()
            fig_elbow_alfonso .add_trace(go.Scatter(x=list(k_range), y=inertias, mode='lines+markers'))
            fig_elbow_alfonso .add_vline(x=optimal_k, line_width=2, line_dash='dash', line_color='red')
            fig_elbow_alfonso .update_layout(title=f"Elbow Method - Optimal k: {optimal_k}",
                            xaxis_title="Number of Clusters (k)",
                            yaxis_title="Inertia")

            st.plotly_chart(fig_elbow_alfonso)
            
            
########### Open streetmap ######################
            
        df = pd.read_csv("data/Alfonso/ALFONSO 2020 - 2024.csv")

    # Drop rows with missing coordinates
        df = df.dropna(subset=["Latitude", "Longitude"])

    # Optional: Reset index
        df = df.reset_index(drop=True)


        k = 4 # d=2+1

    # Prepare data for clustering
        coords = df[['Latitude', 'Longitude']]

    # Fit KMeans
        kmeans = KMeans(n_clusters=k, random_state=0, n_init='auto')
        df['Cluster'] = kmeans.fit_predict(coords)

        custom_colors = px.colors.qualitative.Bold

        # Plot with Plotly
        fig_alfonso = px.scatter_mapbox(
        df,
        lat='Latitude',
        lon='Longitude',
        color='Cluster',
        zoom=10,
        mapbox_style='open-street-map',
        title=f'K-Means Clustering of Road Accidents (k={k})'
        )

        st.plotly_chart(fig_alfonso)


        
        
################## K means clustering based on number of accidents ####################
        df = pd.read_csv("data/Alfonso/ALFONSO total.csv")
        
        X = df[['Number of Accidents']]
        inertia = []
        K = range(1, 8)

        for k in K:
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(X)
            inertia.append(kmeans.inertia_)

        knee = KneeLocator(K, inertia, curve='convex', direction='decreasing')
        optimal_k = knee.knee

        # Plot Elbow Method
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=list(K), y=inertia, mode='lines+markers', name='Inertia'))
        fig.add_vline(x=optimal_k, line_dash='dash', line_color='red',
                    annotation_text=f"Elbow at k={optimal_k}", annotation_position="top right")

        fig.update_layout(
            title="Elbow Method for Optimal k (Interactive)",
            xaxis_title="Number of Clusters (k)",
            yaxis_title="Inertia",
            hovermode='x unified',
        )
        st.subheader("Elbow Method to Determine Optimal k (Interactive)")
        st.plotly_chart(fig)
        st.success(f"✅ Automatically detected optimal number of clusters: **k = {optimal_k}**")

        # --- KMeans Clustering ---
        kmeans = KMeans(n_clusters=optimal_k, random_state=42)
        df['Cluster'] = kmeans.fit_predict(X)

        # Custom color map for clusters
        cluster_color_map = {
            0: 'green',
            1: 'red',
            2: 'yellow'
            # Add more if needed
        }

        # Sort for better visuals
        plot_df = df.sort_values('Number of Accidents')

        # Plot using Plotly
        fig2 = go.Figure()

        for cluster_num in sorted(df['Cluster'].unique()):
            cluster_data = plot_df[plot_df['Cluster'] == cluster_num]
            color = cluster_color_map.get(cluster_num, 'gray')  # fallback to gray
            fig2.add_trace(go.Scatter(
                x=cluster_data['Number of Accidents'],
                y=cluster_data['Barangay'],
                mode='markers',
                marker=dict(
                    color=color,
                    size=8,
                    symbol='x'
                ),
                name=f"Cluster {cluster_num}"
            ))

        fig2.update_layout(
            title="KMeans Clustering of Barangays based on Number of Accidents",
            xaxis_title="Number of Accidents",
            yaxis_title="Barangay",
            template="plotly_white",
            width=800,
            height=800,
        )

        st.subheader("KMeans Clustering of Barangays based on Number of Accidents")
        st.plotly_chart(fig2, use_container_width=True)
                
        
        
##################### DBSCAN ###########################        
        
    elif tab == "DBSCAN":
        
        df = pd.read_csv("data/Alfonso/ALFONSO 2020 - 2024.csv")
        coords = df[['Latitude', 'Longitude']]

        # Streamlit sliders
        eps = st.slider("Epsilon (distance threshold)", 0.01, 5.0, 0.5, step=0.01)
        min_samples = st.slider("Min Samples (points per cluster)", 1, 20, 5)

        # Standardize coordinates
        scaler = StandardScaler()
        coords_scaled = scaler.fit_transform(coords)

        # Apply DBSCAN
        db = DBSCAN(eps=eps, min_samples=min_samples)
        df['cluster'] = db.fit_predict(coords_scaled)

        # ✅ Compute silhouette score only if ≥2 clusters (excluding noise)
        labels = df['cluster']
        n_clusters = len(set(labels)) - (1 if -1 in labels.values else 0)

        if n_clusters >= 2:
            sil_score = silhouette_score(coords_scaled, labels)
            st.success(f"Silhouette Score: **{sil_score:.4f}**")
        else:
            st.warning("Silhouette score requires at least 2 clusters (excluding noise).")

        # Create Folium map
        m = folium.Map(location=[df['Latitude'].mean(), df['Longitude'].mean()], zoom_start=12)

        # Define colors for clusters
        cluster_colors = ['red', 'blue', 'green', 'purple', 'orange', 'darkred', 'lightblue', 'darkgreen', 'gray']
        noise_color = 'black'

        for _, row in df.iterrows():
            cluster = row['cluster']
            color = noise_color if cluster == -1 else cluster_colors[cluster % len(cluster_colors)]
            folium.CircleMarker(
                location=[row['Latitude'], row['Longitude']],
                radius=5,
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.7,
                popup=f"Cluster: {cluster}"
            ).add_to(m)

        # Show map
        st.subheader("Interactive Cluster Map")
        st_folium(m, width=700, height=500)

        
        
        
                
                
######################### GMA ###############################

elif page == "GMA":


    st.write("Clustering in GMA")
    
    tab = st.radio("Select Clustering", ["K-Means", "DBSCAN"], key="gma")


    if tab == "K-Means":
        df = pd.read_csv("data/GMA/GMA 2020 - 2024.csv")

        # Drop rows with missing coordinates
        df = df.dropna(subset=["Latitude", "Longitude"])

    # Optional: Reset index
        df = df.reset_index(drop=True)


    # Select number of clusters (or use Elbow method to determine)
        k = st.slider("Select number of clusters (k)", min_value=1, max_value=10, value=3, key="slider2")

    # Prepare data for clustering
        coords = df[['Latitude', 'Longitude']]

    # Fit KMeans
        kmeans = KMeans(n_clusters=k, random_state=0, n_init='auto')
        df['Cluster'] = kmeans.fit_predict(coords)

    # Plot with Plotly
        fig_gma = px.scatter_mapbox(
            df,
        lat='Latitude',
        lon='Longitude',
        color='Cluster',
        zoom=11.5,
        mapbox_style='open-street-map',
        title=f'K-Means Clustering of Road Accidents (k={k})'
        )

        st.plotly_chart(fig_gma)


    # ELBOW METHOD

        df = pd.read_csv("data/GMA/GMA 2020 - 2024.csv")

    # Drop rows with missing lat/lon
        df = df.dropna(subset=['Latitude', 'Longitude'])

    # Use only coordinates for clustering
        coords = df[['Latitude', 'Longitude']]

    # Elbow method: compute inertia for a range of k
        inertias = []
        k_range = range(1, 11)

        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=0, n_init='auto')
            kmeans.fit(coords)
            inertias.append(kmeans.inertia_)

    # Plot the Elbow chart
        fig_elbow_gma = go.Figure()
        fig_elbow_gma .add_trace(go.Scatter(x=list(k_range), y=inertias, mode='lines+markers'))
        fig_elbow_gma .update_layout(title="Elbow Method for Optimal k",
                            xaxis_title="Number of Clusters (k)",
                            yaxis_title="Inertia (Within-cluster sum of squares)")

        st.plotly_chart(fig_elbow_gma)



    ##### Automation of the elbow method

    # Load and clean data
        df = pd.read_csv("data/GMA/GMA 2020 - 2024.csv")
        df = df.dropna(subset=['Latitude', 'Longitude'])
        coords = df[['Latitude', 'Longitude']]

    # Compute inertia for k = 1 to 10
        inertias = []
        k_range = range(1, 11)
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=0, n_init='auto')
            kmeans.fit(coords)
            inertias.append(kmeans.inertia_)

    # Use kneed to detect the elbow
        knee = KneeLocator(k_range, inertias, curve='convex', direction='decreasing')
        optimal_k = knee.elbow

    # Show Elbow Plot
        fig_elbow_gma  = go.Figure()
        fig_elbow_gma .add_trace(go.Scatter(x=list(k_range), y=inertias, mode='lines+markers'))
        fig_elbow_gma .add_vline(x=optimal_k, line_width=2, line_dash='dash', line_color='red')
        fig_elbow_gma .update_layout(title=f"Elbow Method - Optimal k: {optimal_k}",
                            xaxis_title="Number of Clusters (k)",
                            yaxis_title="Inertia")

        st.plotly_chart(fig_elbow_gma)
        
        
        ################## K means clustering based on number of accidents ####################
        df = pd.read_csv("data/GMA/GMA total.csv")
        
        X = df[['Number of Accidents']]
        inertia = []
        K = range(1, 8)

        for k in K:
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(X)
            inertia.append(kmeans.inertia_)

        knee = KneeLocator(K, inertia, curve='convex', direction='decreasing')
        optimal_k = knee.knee

        # Plot Elbow Method
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=list(K), y=inertia, mode='lines+markers', name='Inertia'))
        fig.add_vline(x=optimal_k, line_dash='dash', line_color='red',
                    annotation_text=f"Elbow at k={optimal_k}", annotation_position="top right")

        fig.update_layout(
            title="Elbow Method for Optimal k (Interactive)",
            xaxis_title="Number of Clusters (k)",
            yaxis_title="Inertia",
            hovermode='x unified'
        )
        st.subheader("Elbow Method to Determine Optimal k (Interactive)")
        st.plotly_chart(fig)
        st.success(f"✅ Automatically detected optimal number of clusters: **k = {optimal_k}**")

        # --- KMeans Clustering ---
        kmeans = KMeans(n_clusters=optimal_k, random_state=42)
        df['Cluster'] = kmeans.fit_predict(X)

        # Custom color map by cluster index
        cluster_color_map = {
            0: 'yellow',
            1: 'red',
            2: 'green'
            # Add more if k > 3, e.g., 3: 'blue', 4: 'purple', etc.
        }

        # Sort for better visuals
        plot_df = df.sort_values('Number of Accidents')

        # Plot using Plotly
        fig2 = go.Figure()

        for cluster_num in sorted(df['Cluster'].unique()):
            cluster_data = plot_df[plot_df['Cluster'] == cluster_num]
            color = cluster_color_map.get(cluster_num, 'gray')  # Default to gray if not mapped
            fig2.add_trace(go.Scatter(
                x=cluster_data['Number of Accidents'],
                y=cluster_data['Barangay'],
                mode='markers',
                marker=dict(
                    color=color,
                    size=8,
                    symbol='x'
                ),
                name=f"Cluster {cluster_num}"
            ))

        fig2.update_layout(
            title="KMeans Clustering of Barangays based on Number of Accidents",
            xaxis_title="Number of Accidents",
            yaxis_title="Barangay",
            template="plotly_white",
            width=800,
            height=800
        )

        st.subheader("KMeans Clustering of Barangays based on Number of Accidents")
        st.plotly_chart(fig2, use_container_width=True)
        
#################################### DBSCAN ####################################


    elif tab == "DBSCAN":
        
        df = pd.read_csv("data/GMA/GMA 2020 - 2024.csv")

        df = df.dropna(subset=["Latitude", "Longitude"])
        coords = df[['Latitude', 'Longitude']].values

        # Streamlit UI
        st.subheader("DBSCAN Clustering")

        eps_meters = st.slider("Epsilon (meters)", 50, 1000, 300, step=50)
        min_sample = st.slider("Min Samples", 2, 20, 4)

        # DBSCAN
        coords_rad = np.radians(coords)
        eps_rad = eps_meters / 6371000
        db = DBSCAN(eps=eps_rad, min_samples=min_sample, metric='haversine')
        df['cluster'] = db.fit_predict(coords_rad)

        # Split data
        clusters = df[df['cluster'] != -1]
        noise = df[df['cluster'] == -1]

        # Define custom colors (as many as you need)
        custom_colors = [
            '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
            '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
        ]

        # Create Plotly Map
        fig = go.Figure()

        # Plot clusters
        unique_clusters = clusters['cluster'].unique()
        for i, cluster_id in enumerate(unique_clusters):
            cluster_data = clusters[clusters['cluster'] == cluster_id]
            fig.add_trace(go.Scattermapbox(
                lat=cluster_data['Latitude'],
                lon=cluster_data['Longitude'],
                mode='markers',
                marker=dict(size=8, color=custom_colors[i % len(custom_colors)]),
                name=f'Cluster {cluster_id}',
                hoverinfo='text',
                text=[f'Cluster {cluster_id}'] * len(cluster_data)
            ))

        # Plot noise as black X
        if not noise.empty:
            fig.add_trace(go.Scattermapbox(
                lat=noise['Latitude'],
                lon=noise['Longitude'],
                mode='markers',
                marker=dict(size=8, color='black'),
                name='Noise',
                hoverinfo='text',
                text=['Noise'] * len(noise)
            ))

        fig.update_layout(
            mapbox_style="open-street-map",
            mapbox_zoom=11,
            mapbox_center={"lat": df["Latitude"].mean(), "lon": df["Longitude"].mean()},
            margin={"r":0,"t":40,"l":0,"b":0},
            title="🗺️ DBSCAN Clustering"
        )

        st.plotly_chart(fig)

        # Stats
        n_clusters = len(unique_clusters)
        n_noise = len(noise)
        st.write(f"✅ Detected Clusters: {n_clusters}")
        st.write(f"❌ Noise Points: {n_noise}")
                        
        
elif page == "Carmona":

###################### Carmona ######################


    st.write("Clustering in Carmona")

    tab = st.radio("Select Clustering", ["K-Means", "DBSCAN"], key="carmona")
    
    if tab == "K-Means":
    
        df = pd.read_csv("data/Carmona/CARMONA 2020 - 2024.csv")

    # Drop rows with missing coordinates
        df = df.dropna(subset=["Latitude", "Longitude"])

    # Optional: Reset index
        df = df.reset_index(drop=True)


    # Select number of clusters (or use Elbow method to determine)
        k = st.slider("Select number of clusters (k)", min_value=1, max_value=10, value=3, key="slider3")

    # Prepare data for clustering
        coords = df[['Latitude', 'Longitude']]

    # Fit KMeans
        kmeans = KMeans(n_clusters=k, random_state=0, n_init='auto')
        df['Cluster'] = kmeans.fit_predict(coords)

    # Plot with Plotly
        fig_gma = px.scatter_mapbox(
            df,
            lat='Latitude',
            lon='Longitude',
            color='Cluster',
            zoom=11.5,
            mapbox_style='open-street-map',
            title=f'K-Means Clustering of Road Accidents (k={k})'
        )

        st.plotly_chart(fig_gma)


    # ELBOW METHOD

        df = pd.read_csv("data/Carmona/CARMONA 2020 - 2024.csv")

    # Drop rows with missing lat/lon
        df = df.dropna(subset=['Latitude', 'Longitude'])

    # Use only coordinates for clustering
        coords = df[['Latitude', 'Longitude']]

    # Elbow method: compute inertia for a range of k
        inertias = []
        k_range = range(1, 11)

        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=0, n_init='auto')
            kmeans.fit(coords)
            inertias.append(kmeans.inertia_)

    # Plot the Elbow chart
        fig_elbow_gma = go.Figure()
        fig_elbow_gma .add_trace(go.Scatter(x=list(k_range), y=inertias, mode='lines+markers'))
        fig_elbow_gma .update_layout(title="Elbow Method for Optimal k",
                            xaxis_title="Number of Clusters (k)",
                            yaxis_title="Inertia (Within-cluster sum of squares)")

        st.plotly_chart(fig_elbow_gma)



    ##### Automation of the elbow method

    # Load and clean data
        df = pd.read_csv("data/Carmona/CARMONA 2020 - 2024.csv")
        df = df.dropna(subset=['Latitude', 'Longitude'])
        coords = df[['Latitude', 'Longitude']]

    # Compute inertia for k = 1 to 10
        inertias = []
        k_range = range(1, 11)
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=0, n_init='auto')
            kmeans.fit(coords)
            inertias.append(kmeans.inertia_)

    # Use kneed to detect the elbow
        knee = KneeLocator(k_range, inertias, curve='convex', direction='decreasing')
        optimal_k = knee.elbow

    # Show Elbow Plot
        fig_elbow_gma  = go.Figure()
        fig_elbow_gma .add_trace(go.Scatter(x=list(k_range), y=inertias, mode='lines+markers'))
        fig_elbow_gma .add_vline(x=optimal_k, line_width=2, line_dash='dash', line_color='red')
        fig_elbow_gma .update_layout(title=f"Elbow Method - Optimal k: {optimal_k}",
                            xaxis_title="Number of Clusters (k)",
                            yaxis_title="Inertia")

        st.plotly_chart(fig_elbow_gma)
        
        
        ################## K means clustering based on number of accidents ####################
        df = pd.read_csv("data/Carmona/CARMONA total.csv")
        
        X = df[['Number of Accidents']]
        inertia = []
        K = range(1, 8)

        for k in K:
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(X)
            inertia.append(kmeans.inertia_)

        knee = KneeLocator(K, inertia, curve='convex', direction='decreasing')
        optimal_k = knee.knee

        # Plot Elbow Method
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=list(K), y=inertia, mode='lines+markers', name='Inertia'))
        fig.add_vline(x=optimal_k, line_dash='dash', line_color='red',
                    annotation_text=f"Elbow at k={optimal_k}", annotation_position="top right")

        fig.update_layout(
            title="Elbow Method for Optimal k (Interactive)",
            xaxis_title="Number of Clusters (k)",
            yaxis_title="Inertia",
            hovermode='x unified'
        )
        st.subheader("Elbow Method to Determine Optimal k (Interactive)")
        st.plotly_chart(fig)
        st.success(f"✅ Automatically detected optimal number of clusters: **k = {optimal_k}**")

        # --- KMeans Clustering ---
        kmeans = KMeans(n_clusters=optimal_k, random_state=42)
        df['Cluster'] = kmeans.fit_predict(X)

        # Custom color map by cluster index
        cluster_color_map = {
            0: 'yellow',
            1: 'red',
            2: 'green'
            # Add more if k > 3, e.g., 3: 'blue', 4: 'purple', etc.
        }

        # Sort for better visuals
        plot_df = df.sort_values('Number of Accidents')

        # Plot using Plotly
        fig2 = go.Figure()

        for cluster_num in sorted(df['Cluster'].unique()):
            cluster_data = plot_df[plot_df['Cluster'] == cluster_num]
            color = cluster_color_map.get(cluster_num, 'gray')  # Default to gray if not mapped
            fig2.add_trace(go.Scatter(
                x=cluster_data['Number of Accidents'],
                y=cluster_data['Barangay'],
                mode='markers',
                marker=dict(
                    color=color,
                    size=8,
                    symbol='x'
                ),
                name=f"Cluster {cluster_num}"
            ))

        fig2.update_layout(
            title="KMeans Clustering of Barangays based on Number of Accidents",
            xaxis_title="Number of Accidents",
            yaxis_title="Barangay",
            template="plotly_white"
        )

        st.subheader("KMeans Clustering of Barangays based on Number of Accidents")
        st.plotly_chart(fig2, use_container_width=True)

################################### DBSCAN ####################################
        
    elif tab == "DBSCAN":
    
        df = pd.read_csv("data/Carmona/CARMONA 2020 - 2024.csv")

        df = df.dropna(subset=["Latitude", "Longitude"])
        coords = df[['Latitude', 'Longitude']].values

        # Streamlit UI
        st.subheader("DBSCAN Clustering")

        eps_meters = st.slider("Epsilon (meters)", 50, 1000, 300, step=50)
        min_sample = st.slider("Min Samples", 2, 20, 4)

        # DBSCAN
        coords_rad = np.radians(coords)
        eps_rad = eps_meters / 6371000
        db = DBSCAN(eps=eps_rad, min_samples=min_sample, metric='haversine')
        df['cluster'] = db.fit_predict(coords_rad)

        # Split data
        clusters = df[df['cluster'] != -1]
        noise = df[df['cluster'] == -1]

        # Define custom colors (as many as you need)
        custom_colors = [
            '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
            '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
        ]

        # Create Plotly Map
        fig = go.Figure()

        # Plot clusters
        unique_clusters = clusters['cluster'].unique()
        for i, cluster_id in enumerate(unique_clusters):
            cluster_data = clusters[clusters['cluster'] == cluster_id]
            fig.add_trace(go.Scattermapbox(
                lat=cluster_data['Latitude'],
                lon=cluster_data['Longitude'],
                mode='markers',
                marker=dict(size=8, color=custom_colors[i % len(custom_colors)]),
                name=f'Cluster {cluster_id}',
                hoverinfo='text',
                text=[f'Cluster {cluster_id}'] * len(cluster_data)
            ))

        # Plot noise as black X
        if not noise.empty:
            fig.add_trace(go.Scattermapbox(
                lat=noise['Latitude'],
                lon=noise['Longitude'],
                mode='markers',
                marker=dict(size=8, color='black'),
                name='Noise',
                hoverinfo='text',
                text=['Noise'] * len(noise)
            ))

        fig.update_layout(
            mapbox_style="open-street-map",
            mapbox_zoom=11,
            mapbox_center={"lat": df["Latitude"].mean(), "lon": df["Longitude"].mean()},
            margin={"r":0,"t":40,"l":0,"b":0},
            title="🗺️ DBSCAN Clustering"
        )

        st.plotly_chart(fig)

        # Stats
        n_clusters = len(unique_clusters)
        n_noise = len(noise)
        st.write(f"✅ Detected Clusters: {n_clusters}")
        st.write(f"❌ Noise Points: {n_noise}")