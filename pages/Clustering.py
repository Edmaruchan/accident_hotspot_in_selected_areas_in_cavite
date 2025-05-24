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




st.title("Clustering")

page = st.radio("Select Area", ["Alfonso", "GMA", "Carmona"])


########################### ALFONSO ####################
if page == "Alfonso":

    st.write("Clustering in Alfonso")
    
    tab = st.radio("Select Clustering", ["K-Means", "DBSCAN"], key="alfonso")

    if tab == "K-Means":

        st.subheader("K-Means Clustering")
            
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


        # ELBOW METHOD

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
        
    elif tab == "DBSCAN":
        
        df = pd.read_csv("data/Alfonso/ALFONSO 2020 - 2024.csv")

        coords_deg = df[['Latitude', 'Longitude']].dropna().values
        
        coords_rad = np.radians(coords_deg)

        # Streamlit slider to set min_samples
        min_samples = 3 # 2+1

        # Compute k-distance
        k = min_samples - 1
        neighbors = NearestNeighbors(n_neighbors=min_samples, metric='haversine')
        neighbors_fit = neighbors.fit(coords_rad)
        distances, indices = neighbors_fit.kneighbors(coords_rad)
        k_distances = np.sort(distances[:, k]) * 6371000  # Convert radians to meters

        # Detect elbow using KneeLocator
        kneedle = KneeLocator(
            x=range(len(k_distances)),
            y=k_distances,
            curve='convex',
            direction='increasing'
        )
        elbow_index = kneedle.knee
        elbow_eps = k_distances[elbow_index] if elbow_index is not None else None

        # Create interactive Plotly line chart
        k_dist_df = pd.DataFrame({
            'Point Index (Sorted)': np.arange(len(k_distances)),
            'k-Distance (meters)': k_distances
        })

        fig = px.line(
            k_dist_df,
            x='Point Index (Sorted)',
            y='k-Distance (meters)',
            title='k-Distance Plot with Elbow Detection',
            markers=True
        )

        # Mark the elbow point on the plot
        if elbow_index is not None:
            fig.add_scatter(
                x=[elbow_index],
                y=[elbow_eps],
                mode='markers+text',
                marker=dict(color='red', size=10),
                text=[f"Elbow @ {elbow_eps:.2f}m"],
                textposition='top right',
                name='Elbow Point'
            )
            st.success(f"Estimated optimal `eps` (in meters): **{elbow_eps:.2f}**")
        else:
            st.warning("No elbow point detected.")

        st.plotly_chart(fig)
        
        
        df = pd.read_csv("data/Alfonso/ALFONSO 2020 - 2024.csv")
        
        # Load data
        df = pd.read_csv("data/Alfonso/ALFONSO 2020 - 2024.csv")
        df = df.dropna(subset=["Latitude", "Longitude"])

        coords = df[['Latitude', 'Longitude']].values

        st.subheader("DBSCAN Clustering")

        eps_meters = st.slider("Epsilon (meters)", 0.1, 0.3, 0.9, step=0.1)
        min_sample = 3
        

        # DBSCAN parameters
        coords_rad = np.radians(coords)
        #eps_rad = eps_meters / 6371.0  # Earth radius in km

        
        eps_rad = eps_meters / 6371.0  # 300 meters in radians
        db = DBSCAN(eps=eps_rad, min_samples=min_sample, metric='haversine')
        df['cluster'] = db.fit_predict(coords_rad)

        # Separate clusters and noise
        clusters = df[df['cluster'] != -1]
        noise = df[df['cluster'] == -1]

        # Plotting
        fig, ax = plt.subplots(figsize=(8, 6))

        # Plot clusters
        unique_clusters = clusters['cluster'].unique()
        colors = plt.cm.tab20(np.linspace(0, 1, len(unique_clusters)))
        handles = []

        for i, cluster_id in enumerate(unique_clusters):
            cluster_data = clusters[clusters['cluster'] == cluster_id]
            ax.scatter(cluster_data['Longitude'], cluster_data['Latitude'], 
                    c=[colors[i]], s=10, label=f'Cluster {cluster_id}')
            handles.append(mpatches.Patch(color=colors[i], label=f'Cluster {cluster_id}'))

        # Plot noise points
        if not noise.empty:
            ax.scatter(noise['Longitude'], noise['Latitude'], 
                    c='gray', s=10, marker='x', label='Noise')
            handles.append(mpatches.Patch(color='gray', label='Noise'))

        # Legend
        ax.legend(handles=handles, title="Clusters", loc='upper right', fontsize=8, title_fontsize=9)

        # Axis labels and grid
        ax.set_title("DBSCAN Clustering")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.grid(True)

        # Show in Streamlit
        st.pyplot(fig)
        
        st.write(f"Detected Clusters: {df['cluster'].nunique() - (1 if -1 in df['cluster'].unique() else 0)}")
        st.write(f"Noise Points: {(df['cluster'] == -1).sum()}")
        
        ###########################
        df = pd.read_csv("data/Alfonso/ALFONSO total.csv")
        
                # Select number of clusters
        k = st.slider("Select number of clusters (K)", min_value=1, max_value=10, value=3)

        # Perform KMeans clustering
        X = df[['Number of Accidents']]
        kmeans = KMeans(n_clusters=k, random_state=42)
        df['Cluster'] = kmeans.fit_predict(X)

        # Display clustered data
        st.subheader("Clustered Data")
        st.dataframe(df)

        # Plot the clusters
        fig, ax = plt.subplots()
        colors = ['red', 'blue', 'green', 'purple', 'orange', 'cyan', 'magenta', 'yellow', 'gray', 'brown']
        for cluster in range(k):
            cluster_data = df[df['Cluster'] == cluster]
            ax.scatter(cluster_data['Barangay'], cluster_data['Number of Accidents'],
                    color=colors[cluster % len(colors)],
                    label=f'Cluster {cluster}', s=100)

        ax.set_xlabel("Barangay")
        ax.set_ylabel("Number of Accidents")
        ax.set_title("KMeans Clustering of Barangays")
        ax.tick_params(axis='x', rotation=45)
        ax.legend()
        st.pyplot(fig)
        
        # Elbow method to determine optimal k
        inertia = []
        K = range(1, 10)
        for k in K:
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(X)
            inertia.append(kmeans.inertia_)

        fig, ax = plt.subplots()
        ax.plot(K, inertia, 'bx-')
        ax.set_xlabel('Number of clusters (k)')
        ax.set_ylabel('Inertia')
        ax.set_title('Elbow Method For Optimal k')

        st.subheader("Elbow Method to Determine Optimal k")
        st.pyplot(fig)
        

elif page == "GMA":


######################### GMA ###############################


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
        
        
    elif tab == "DBSCAN":
        
        df = pd.read_csv("data/GMA/GMA 2020 - 2024.csv")

        coords_deg = df[['Latitude', 'Longitude']].dropna().values
        
        coords_rad = np.radians(coords_deg)

        # Streamlit slider to set min_samples
        min_samples = st.slider("Select min_samples (for DBSCAN)", min_value=2, max_value=10, value=4)

        # Compute k-distance
        k = min_samples - 1
        neighbors = NearestNeighbors(n_neighbors=min_samples, metric='haversine')
        neighbors_fit = neighbors.fit(coords_rad)
        distances, indices = neighbors_fit.kneighbors(coords_rad)
        k_distances = np.sort(distances[:, k]) * 6371000  # Convert radians to meters

        # Detect elbow using KneeLocator
        kneedle = KneeLocator(
            x=range(len(k_distances)),
            y=k_distances,
            curve='convex',
            direction='increasing'
        )
        elbow_index = kneedle.knee
        elbow_eps = k_distances[elbow_index] if elbow_index is not None else None

        # Create interactive Plotly line chart
        k_dist_df = pd.DataFrame({
            'Point Index (Sorted)': np.arange(len(k_distances)),
            'k-Distance (meters)': k_distances
        })

        fig = px.line(
            k_dist_df,
            x='Point Index (Sorted)',
            y='k-Distance (meters)',
            title='k-Distance Plot with Elbow Detection',
            markers=True
        )

        # Mark the elbow point on the plot
        if elbow_index is not None:
            fig.add_scatter(
                x=[elbow_index],
                y=[elbow_eps],
                mode='markers+text',
                marker=dict(color='red', size=10),
                text=[f"Elbow @ {elbow_eps:.2f}m"],
                textposition='top right',
                name='Elbow Point'
            )
            st.success(f"Estimated optimal `eps` (in meters): **{elbow_eps:.2f}**")
        else:
            st.warning("No elbow point detected.")

        st.plotly_chart(fig)
        
        
        df = pd.read_csv("data/GMA/GMA 2020 - 2024.csv")
        
        df = df.dropna(subset=["Latitude", "Longitude"])

        coords = df[['Latitude', 'Longitude']].values

        st.subheader("DBSCAN Clustering")

        eps_meters = st.slider("Epsilon (meters)", 0.1, 0.3, 0.9, step=0.1)
        min_sample = st.slider("Min Samples", 2, 20, 4)
        

        # DBSCAN parameters
        coords_rad = np.radians(coords)
        #eps_rad = eps_meters / 6371.0  # Earth radius in km

        
        eps_rad = eps_meters / 6371.0  # 300 meters in radians
        db = DBSCAN(eps=eps_rad, min_samples=min_sample, metric='haversine')
        df['cluster'] = db.fit_predict(coords_rad)

        # Separate clusters and noise
        clusters = df[df['cluster'] != -1]
        noise = df[df['cluster'] == -1]

        # Plotting
        fig, ax = plt.subplots(figsize=(8, 6))

        # Plot clusters
        unique_clusters = clusters['cluster'].unique()
        colors = plt.cm.tab20(np.linspace(0, 1, len(unique_clusters)))
        handles = []

        for i, cluster_id in enumerate(unique_clusters):
            cluster_data = clusters[clusters['cluster'] == cluster_id]
            ax.scatter(cluster_data['Longitude'], cluster_data['Latitude'], 
                    c=[colors[i]], s=10, label=f'Cluster {cluster_id}')
            handles.append(mpatches.Patch(color=colors[i], label=f'Cluster {cluster_id}'))

        # Plot noise points
        if not noise.empty:
            ax.scatter(noise['Longitude'], noise['Latitude'], 
                    c='gray', s=10, marker='x', label='Noise')
            handles.append(mpatches.Patch(color='gray', label='Noise'))

        # Legend
        ax.legend(handles=handles, title="Clusters", loc='lower right', fontsize=8, title_fontsize=9)

        # Axis labels and grid
        ax.set_title("DBSCAN Clustering")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.grid(True)

        # Show in Streamlit
        st.pyplot(fig)
        
        st.write(f"Detected Clusters: {df['cluster'].nunique() - (1 if -1 in df['cluster'].unique() else 0)}")
        st.write(f"Noise Points: {(df['cluster'] == -1).sum()}")
                        

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
        
        
    elif tab == "DBSCAN":
    
        df = pd.read_csv("data/Carmona/CARMONA 2020 - 2024.csv")

        coords_deg = df[['Latitude', 'Longitude']].dropna().values
        
        coords_rad = np.radians(coords_deg)

        # Streamlit slider to set min_samples
        min_samples = st.slider("Select min_samples (for DBSCAN)", min_value=2, max_value=10, value=4)

        # Compute k-distance
        k = min_samples - 1
        neighbors = NearestNeighbors(n_neighbors=min_samples, metric='haversine')
        neighbors_fit = neighbors.fit(coords_rad)
        distances, indices = neighbors_fit.kneighbors(coords_rad)
        k_distances = np.sort(distances[:, k]) * 6371000  # Convert radians to meters

        # Detect elbow using KneeLocator
        kneedle = KneeLocator(
            x=range(len(k_distances)),
            y=k_distances,
            curve='convex',
            direction='increasing'
        )
        elbow_index = kneedle.knee
        elbow_eps = k_distances[elbow_index] if elbow_index is not None else None

        # Create interactive Plotly line chart
        k_dist_df = pd.DataFrame({
            'Point Index (Sorted)': np.arange(len(k_distances)),
            'k-Distance (meters)': k_distances
        })

        fig = px.line(
            k_dist_df,
            x='Point Index (Sorted)',
            y='k-Distance (meters)',
            title='k-Distance Plot with Elbow Detection',
            markers=True
        )

        # Mark the elbow point on the plot
        if elbow_index is not None:
            fig.add_scatter(
                x=[elbow_index],
                y=[elbow_eps],
                mode='markers+text',
                marker=dict(color='red', size=10),
                text=[f"Elbow @ {elbow_eps:.2f}m"],
                textposition='top right',
                name='Elbow Point'
            )
            st.success(f"Estimated optimal `eps` (in meters): **{elbow_eps:.2f}**")
        else:
            st.warning("No elbow point detected.")

        st.plotly_chart(fig)
        
        
        df = pd.read_csv("data/Carmona/CARMONA 2020 - 2024.csv")
        
        df = df.dropna(subset=["Latitude", "Longitude"])

        coords = df[['Latitude', 'Longitude']].values

        st.subheader("DBSCAN Clustering")

        eps_meters = st.slider("Epsilon (meters)", 0.1, 0.3, 0.9, step=0.1)
        min_sample = st.slider("Min Samples", 2, 20, 4)
        

        # DBSCAN parameters
        coords_rad = np.radians(coords)
        #eps_rad = eps_meters / 6371.0  # Earth radius in km

        
        eps_rad = eps_meters / 6371.0  # 300 meters in radians
        db = DBSCAN(eps=eps_rad, min_samples=min_sample, metric='haversine')
        df['cluster'] = db.fit_predict(coords_rad)

        # Separate clusters and noise
        clusters = df[df['cluster'] != -1]
        noise = df[df['cluster'] == -1]

        # Plotting
        fig, ax = plt.subplots(figsize=(8, 6))

        # Plot clusters
        unique_clusters = clusters['cluster'].unique()
        colors = plt.cm.tab20(np.linspace(0, 1, len(unique_clusters)))
        handles = []

        for i, cluster_id in enumerate(unique_clusters):
            cluster_data = clusters[clusters['cluster'] == cluster_id]
            ax.scatter(cluster_data['Longitude'], cluster_data['Latitude'], 
                    c=[colors[i]], s=10, label=f'Cluster {cluster_id}')
            handles.append(mpatches.Patch(color=colors[i], label=f'Cluster {cluster_id}'))

        # Plot noise points
        if not noise.empty:
            ax.scatter(noise['Longitude'], noise['Latitude'], 
                    c='gray', s=10, marker='x', label='Noise')
            handles.append(mpatches.Patch(color='gray', label='Noise'))

        # Legend
        ax.legend(handles=handles, title="Clusters", loc='lower right', fontsize=8, title_fontsize=9)

        # Axis labels and grid
        ax.set_title("DBSCAN Clustering")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.grid(True)

        # Show in Streamlit
        st.pyplot(fig)
        
        st.write(f"Detected Clusters: {df['cluster'].nunique() - (1 if -1 in df['cluster'].unique() else 0)}")
        st.write(f"Noise Points: {(df['cluster'] == -1).sum()}")
                        

