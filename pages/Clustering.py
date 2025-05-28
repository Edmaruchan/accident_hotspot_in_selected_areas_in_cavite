import streamlit as st
import pandas as pd
from sklearn.cluster import KMeans
import plotly.express as px
import streamlit as st
from kneed import KneeLocator
import plotly.graph_objects as go






st.title("Clustering")

page = st.sidebar.radio("Select Area", ["Alfonso", "GMA", "Carmona"])

########################### ALFONSO ####################
if page == "Alfonso":
    st.write("Clustering in Alfonso")
    st.subheader("K-Means Clustering")

    col1, col2 = st.columns(2)

    with col1:
        # ELBOW METHOD
        df = pd.read_csv("data/Alfonso/ALFONSO 2020 - 2024.csv")
        df = df.dropna(subset=['Latitude', 'Longitude'])
        coords = df[['Latitude', 'Longitude']]
        inertias = []
        k_range = range(1, 11)
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=0, n_init='auto')
            kmeans.fit(coords)
            inertias.append(kmeans.inertia_)

        fig_elbow_alfonso = go.Figure()
        fig_elbow_alfonso.add_trace(go.Scatter(x=list(k_range), y=inertias, mode='lines+markers'))
        fig_elbow_alfonso.update_layout(title="Elbow Method for Optimal k",
                                        xaxis_title="Number of Clusters (k)",
                                        yaxis_title="Inertia")
        st.plotly_chart(fig_elbow_alfonso)

    with col2:
        # AUTOMATED ELBOW METHOD
        df = pd.read_csv("data/Alfonso/ALFONSO 2020 - 2024.csv")
        df = df.dropna(subset=['Latitude', 'Longitude'])
        coords = df[['Latitude', 'Longitude']]
        inertias = []
        k_range = range(1, 11)
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=0, n_init='auto')
            kmeans.fit(coords)
            inertias.append(kmeans.inertia_)

        knee = KneeLocator(k_range, inertias, curve='convex', direction='decreasing')
        optimal_k = knee.elbow

        fig_elbow_alfonso = go.Figure()
        fig_elbow_alfonso.add_trace(go.Scatter(x=list(k_range), y=inertias, mode='lines+markers'))
        fig_elbow_alfonso.add_vline(x=optimal_k, line_width=2, line_dash='dash', line_color='red')
        fig_elbow_alfonso.update_layout(title=f"Elbow Method - Optimal k: {optimal_k}",
                                        xaxis_title="Number of Clusters (k)",
                                        yaxis_title="Inertia")
        st.plotly_chart(fig_elbow_alfonso)

    # OpenStreetMap Clustering
    df = pd.read_csv("data/Alfonso/ALFONSO 2020 - 2024.csv")
    df = df.dropna(subset=["Latitude", "Longitude"])
    df = df.reset_index(drop=True)
    k = optimal_k
    coords = df[['Latitude', 'Longitude']]
    kmeans = KMeans(n_clusters=k, random_state=0, n_init='auto')
    df['Cluster'] = kmeans.fit_predict(coords)

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

    # KMeans clustering based on number of accidents
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

    kmeans = KMeans(n_clusters=optimal_k, random_state=42)
    df['Cluster'] = kmeans.fit_predict(X)

    cluster_color_map = {
        0: 'green',
        1: 'red',
        2: 'yellow'
    }

    plot_df = df.sort_values('Number of Accidents')
    fig2 = go.Figure()
    for cluster_num in sorted(df['Cluster'].unique()):
        cluster_data = plot_df[plot_df['Cluster'] == cluster_num]
        color = cluster_color_map.get(cluster_num, 'gray')
        fig2.add_trace(go.Scatter(
            x=cluster_data['Number of Accidents'],
            y=cluster_data['Barangay'],
            mode='markers',
            marker=dict(color=color, size=8, symbol='x'),
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

        
        
        
                
                
######################### GMA ###############################
elif page == "GMA":


    st.write("Clustering in GMA")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # ELBOW METHOD
        df = pd.read_csv("data/GMA/GMA 2020 - 2024.csv")
        df = df.dropna(subset=['Latitude', 'Longitude'])
        coords = df[['Latitude', 'Longitude']]
        inertias = []
        k_range = range(1, 11)
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=0, n_init='auto')
            kmeans.fit(coords)
            inertias.append(kmeans.inertia_)

        fig_elbow_gma = go.Figure()
        fig_elbow_gma.add_trace(go.Scatter(x=list(k_range), y=inertias, mode='lines+markers'))
        fig_elbow_gma.update_layout(title="Elbow Method for Optimal k",
                                    xaxis_title="Number of Clusters (k)",
                                    yaxis_title="Inertia")
        st.plotly_chart(fig_elbow_gma)



    with col2:
        # AUTOMATED ELBOW METHOD
        df = pd.read_csv("data/GMA/GMA 2020 - 2024.csv")
        df = df.dropna(subset=['Latitude', 'Longitude'])
        coords = df[['Latitude', 'Longitude']]
        inertias = []
        k_range = range(1, 11)
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=0, n_init='auto')
            kmeans.fit(coords)
            inertias.append(kmeans.inertia_)

        knee = KneeLocator(k_range, inertias, curve='convex', direction='decreasing')
        optimal_k = knee.elbow

        fig_elbow_gma = go.Figure()
        fig_elbow_gma.add_trace(go.Scatter(x=list(k_range), y=inertias, mode='lines+markers'))
        fig_elbow_gma.add_vline(x=optimal_k, line_width=2, line_dash='dash', line_color='red')
        fig_elbow_gma.update_layout(title=f"Elbow Method - Optimal k: {optimal_k}",
                                    xaxis_title="Number of Clusters (k)",
                                    yaxis_title="Inertia")
        st.plotly_chart(fig_elbow_gma)

    

    
    df = pd.read_csv("data/GMA/GMA 2020 - 2024.csv")

        # Drop rows with missing coordinates
    df = df.dropna(subset=["Latitude", "Longitude"])

    # Optional: Reset index
    df = df.reset_index(drop=True)



    k = optimal_k  # Use the optimal k from the elbow method

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
        
        
elif page == "Carmona":

###################### Carmona ######################


    st.write("Clustering in Carmona")

    tab = st.radio("Select Clustering", ["K-Means", "DBSCAN"], key="carmona")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # ELBOW METHOD
        df = pd.read_csv("data/Carmona/CARMONA 2020 - 2024.csv")
        df = df.dropna(subset=['Latitude', 'Longitude'])
        coords = df[['Latitude', 'Longitude']]
        inertias = []
        k_range = range(1, 11)
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=0, n_init='auto')
            kmeans.fit(coords)
            inertias.append(kmeans.inertia_)

        fig_elbow_gma = go.Figure()
        fig_elbow_gma.add_trace(go.Scatter(x=list(k_range), y=inertias, mode='lines+markers'))
        fig_elbow_gma.update_layout(title="Elbow Method for Optimal k",
                                    xaxis_title="Number of Clusters (k)",
                                    yaxis_title="Inertia")
        st.plotly_chart(fig_elbow_gma)



    with col2:
        # AUTOMATED ELBOW METHOD
        df = pd.read_csv("data/Carmona/CARMONA 2020 - 2024.csv")
        df = df.dropna(subset=['Latitude', 'Longitude'])
        coords = df[['Latitude', 'Longitude']]
        inertias = []
        k_range = range(1, 11)
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=0, n_init='auto')
            kmeans.fit(coords)
            inertias.append(kmeans.inertia_)

        knee = KneeLocator(k_range, inertias, curve='convex', direction='decreasing')
        optimal_k = knee.elbow

        fig_elbow_gma = go.Figure()
        fig_elbow_gma.add_trace(go.Scatter(x=list(k_range), y=inertias, mode='lines+markers'))
        fig_elbow_gma.add_vline(x=optimal_k, line_width=2, line_dash='dash', line_color='red')
        fig_elbow_gma.update_layout(title=f"Elbow Method - Optimal k: {optimal_k}",
                                    xaxis_title="Number of Clusters (k)",
                                    yaxis_title="Inertia")
        st.plotly_chart(fig_elbow_gma)

    

    
    df = pd.read_csv("data/Carmona/CARMONA 2020 - 2024.csv")
    df = df.dropna(subset=["Latitude", "Longitude"])

    # Optional: Reset index
    df = df.reset_index(drop=True)



    k = optimal_k  # Use the optimal k from the elbow method

    # Prepare data for clustering
    coords = df[['Latitude', 'Longitude']]

    # Fit KMeans
    kmeans = KMeans(n_clusters=k, random_state=0, n_init='auto')
    df['Cluster'] = kmeans.fit_predict(coords)

    # Plot with Plotly
    fig_carmona = px.scatter_mapbox(
            df,
        lat='Latitude',
        lon='Longitude',
        color='Cluster',
        zoom=11.5,
        mapbox_style='open-street-map',
        title=f'K-Means Clustering of Road Accidents (k={k})'
        )

    st.plotly_chart(fig_carmona)


    
        
        
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
            template="plotly_white",
            width=800,
            height=800
        )

    st.subheader("KMeans Clustering of Barangays based on Number of Accidents")
    st.plotly_chart(fig2, use_container_width=True)