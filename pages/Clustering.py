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
        
        # Elbow method data preparation
        X = df[['Number of Accidents']]
        inertia = []
        K = range(1, 10)

        for k in K:
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(X)
            inertia.append(kmeans.inertia_)

        # Automatically find the elbow point
        knee = KneeLocator(K, inertia, curve='convex', direction='decreasing')
        optimal_k = knee.knee

        # Plot using Plotly
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=list(K), y=inertia, mode='lines+markers', name='Inertia'))
        fig.add_vline(x=optimal_k, line_dash='dash', line_color='red',
                    annotation_text=f"Elbow at k={optimal_k}", annotation_position="top right")

        fig.update_layout(
            title="Elbow Method for Optimal k ",
            xaxis_title="Number of Clusters (k)",
            yaxis_title="Inertia (Within-cluster sum of squares)",
            hovermode='x unified'
        )

        # Display in Streamlit
        st.subheader("K-Means clustering based on Number of Accidents")
        st.plotly_chart(fig)
        st.success(f"✅ Automatically detected optimal number of clusters: **k = {optimal_k}**")
                
        
        k = optimal_k

        # Perform KMeans clustering
        X = df[['Number of Accidents']]
        kmeans = KMeans(n_clusters=k, random_state=42)
        df['Cluster'] = kmeans.fit_predict(X)

        # Plot the clusters
        fig, ax = plt.subplots()
        colors = ['red', 'blue', 'green', 'purple', 'orange', 'cyan', 'magenta', 'yellow', 'gray', 'brown']
        for cluster in range(k):
            cluster_data = df[df['Cluster'] == cluster]
            ax.scatter(cluster_data['Number of Accidents'],cluster_data['Barangay'],
                    color=colors[cluster % len(colors)],
                    label=f'Cluster {cluster}', s=100)

        ax.set_xlabel("Number of Accidents")
        ax.set_ylabel("Barangay")
        ax.set_title("KMeans Clustering of Barangays based on Number of Accidents")
        ax.tick_params(axis='x', rotation=0)
        ax.legend()
        st.pyplot(fig)
        
        
        
##################### DBSCAN ###########################        
        
    elif tab == "DBSCAN":
        
        df = pd.read_csv("data/Alfonso/ALFONSO 2020 - 2024.csv")
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
        
        # Elbow method data preparation
        X = df[['Number of Accidents']]
        inertia = []
        K = range(1, 10)

        for k in K:
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(X)
            inertia.append(kmeans.inertia_)

        # Automatically find the elbow point
        knee = KneeLocator(K, inertia, curve='convex', direction='decreasing')
        optimal_k = knee.knee

        # Plot using Plotly
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=list(K), y=inertia, mode='lines+markers', name='Inertia'))
        fig.add_vline(x=optimal_k, line_dash='dash', line_color='red',
                    annotation_text=f"Elbow at k={optimal_k}", annotation_position="top right")

        fig.update_layout(
            title="Elbow Method for Optimal k (Interactive)",
            xaxis_title="Number of Clusters (k)",
            yaxis_title="Inertia (Within-cluster sum of squares)",
            hovermode='x unified'
        )

        # Display in Streamlit
        st.subheader("Elbow Method to Determine Optimal k (Interactive)")
        st.plotly_chart(fig)
        st.success(f"✅ Automatically detected optimal number of clusters: **k = {optimal_k}**")
                
        
        k = optimal_k

        # Perform KMeans clustering
        X = df[['Number of Accidents']]
        kmeans = KMeans(n_clusters=k, random_state=42)
        df['Cluster'] = kmeans.fit_predict(X)

        # Plot the clusters
        fig, ax = plt.subplots()
        colors = ['red', 'blue', 'green', 'purple', 'orange', 'cyan', 'magenta', 'yellow', 'gray', 'brown']
        for cluster in range(k):
            cluster_data = df[df['Cluster'] == cluster]
            ax.scatter(cluster_data['Number of Accidents'],cluster_data['Barangay'],
                    color=colors[cluster % len(colors)],
                    label=f'Cluster {cluster}', s=100)

        ax.set_xlabel("Number of Accidents")
        ax.set_ylabel("Barangay")
        ax.set_title("KMeans Clustering of Barangays based on Number of Accidents")
        ax.tick_params(axis='x', rotation=0)
        ax.legend()
        st.pyplot(fig)
        
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
        
        # Elbow method data preparation
        X = df[['Number of Accidents']]
        inertia = []
        K = range(1, 8)

        for k in K:
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(X)
            inertia.append(kmeans.inertia_)

        # Automatically find the elbow point
        knee = KneeLocator(K, inertia, curve='convex', direction='decreasing')
        optimal_k = knee.knee

        # Plot using Plotly
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=list(K), y=inertia, mode='lines+markers', name='Inertia'))
        fig.add_vline(x=optimal_k, line_dash='dash', line_color='red',
                    annotation_text=f"Elbow at k={optimal_k}", annotation_position="top right")

        fig.update_layout(
            title="Elbow Method for Optimal k (Interactive)",
            xaxis_title="Number of Clusters (k)",
            yaxis_title="Inertia (Within-cluster sum of squares)",
            hovermode='x unified'
        )

        # Display in Streamlit
        st.subheader("Elbow Method to Determine Optimal k (Interactive)")
        st.plotly_chart(fig)
        st.success(f"✅ Automatically detected optimal number of clusters: **k = {optimal_k}**")
                
        
        k = optimal_k

        # Perform KMeans clustering
        X = df[['Number of Accidents']]
        kmeans = KMeans(n_clusters=k, random_state=42)
        df['Cluster'] = kmeans.fit_predict(X)

        # Plot the clusters
        fig, ax = plt.subplots()
        colors = ['red', 'blue', 'green', 'purple', 'orange', 'cyan', 'magenta', 'yellow', 'gray', 'brown']
        for cluster in range(k):
            cluster_data = df[df['Cluster'] == cluster]
            ax.scatter(cluster_data['Number of Accidents'],cluster_data['Barangay'],
                    color=colors[cluster % len(colors)],
                    label=f'Cluster {cluster}', s=100)

        ax.set_xlabel("Number of Accidents")
        ax.set_ylabel("Barangay")
        ax.set_title("KMeans Clustering of Barangays based on Number of Accidents")
        ax.tick_params(axis='x', rotation=0)
        ax.legend()
        st.pyplot(fig)


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