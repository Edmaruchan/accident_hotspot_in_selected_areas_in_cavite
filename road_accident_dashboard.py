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


st.title("Road Accident Analysis in Cavite")


# Sidebar or top menu
st.subheader("Welcome to the Road Accident Dashboard")
st.write("Select a municipality to view clustering results and maps.")
page = st.radio("Select Area", ["Overview", "Alfonso", "GMA", "Carmona"])

# Pages
if page == "Overview":
  st.write("")
  st.subheader("Road Accident Hotspot Maps")
  df = pd.read_csv("data/ALL.csv")

  # Count the number of accidents per year using the 'year' column directly
  yearly_counts = df['Year'].value_counts().sort_index()

# Convert to DataFrame for display or plotting
  yearly_df = yearly_counts.reset_index()
  yearly_df.columns = ['Year', 'Total Accidents']

# Plot using Plotly
  fig = px.line(
      yearly_df,
     x='Year',
     y='Total Accidents',
     title='Total Road Accidents per Year',
      markers=True
)

  fig.update_layout(xaxis=dict(tickmode='linear'))

# Show in Streamlit
  st.plotly_chart(fig)

  df = pd.read_csv("data/ALL.csv")

# Strip whitespace from column names
  df.columns = df.columns.str.strip()

# Clean Address column: drop NaNs and strip whitespace
  df['Address'] = df['Address'].astype(str).str.strip()

# Count incidents per address
  incident_counts = df['Address'].value_counts().reset_index()
  incident_counts.columns = ['Address', 'Total Incidents']

# Plot interactive horizontal bar chart
  fig = px.bar(
    incident_counts,
    x='Total Incidents',
    y='Address',
    orientation='h',
    title='Total Road Accidents per Barangay (Alfonso, Carmona, GMA) Cavite',
    labels={'Total Incidents': 'Number of Accidents'},
    hover_data={'Total Incidents': True, 'Address': True}
  )

  fig.update_layout(
    yaxis={'categoryorder': 'total ascending'},
    height=1000,         # Adjust height here
    width=900,           # Add custom width
    margin=dict(l=150),  # More left margin for long address labels
  )

  st.plotly_chart(fig, use_container_width=False)

  df = pd.read_csv("data/GMA/GMA 2020 - 2024.csv")

    
  st.title("🟢 Simple k-Distance Plot for DBSCAN")

  df = pd.read_csv("data/ALL.csv")
  df.columns = df.columns.str.lower()

  if 'latitude' in df.columns and 'longitude' in df.columns:
        df = df.dropna(subset=['latitude', 'longitude'])

        # Convert lat/lon to meters using PRS92
        transformer = pyproj.Transformer.from_crs("epsg:4326", "epsg:3123", always_xy=True)
        x_m, y_m = transformer.transform(df['longitude'].values, df['latitude'].values)
        coords = np.column_stack((x_m, y_m))

        # MinPts slider
        k = st.slider("Select min_samples (k)", 2, 20, 5)

        # Compute k-distances
        nn = NearestNeighbors(n_neighbors=k).fit(coords)
        distances, _ = nn.kneighbors(coords)
        k_distances = np.sort(distances[:, k - 1])

        # Plot
        fig, ax = plt.subplots()
        ax.plot(k_distances)
        ax.set_title("k-Distance Plot")
        ax.set_xlabel("Points sorted by distance")
        ax.set_ylabel(f"Distance to {k}th Nearest Neighbor (meters)")
        ax.grid(True)

        st.pyplot(fig)
        st.info("📌 Look for the 'elbow' in the curve to choose your `eps` value.")
  
elif page == "Alfonso":
    st.subheader("Alfonso Analysis")
    #st.image("data/qgis_maps/alfonso.png", caption="Hotspots in Alfonso")

    df = pd.read_csv("data/Alfonso/ALFONSO 2020 - 2024.csv")


# Count the number of accidents per year using the 'year' column directly
    yearly_counts = df['Year'].value_counts().sort_index()

# Convert to DataFrame for display or plotting
    yearly_df = yearly_counts.reset_index()
    yearly_df.columns = ['Year', 'Total Accidents']

# Plot using Plotly
    fig = px.line(
      yearly_df,
     x='Year',
     y='Total Accidents',
     title='Total Road Accidents per Year in Alfonso',
      markers=True
)

    fig.update_layout(xaxis=dict(tickmode='linear'))

# Show in Streamlit
    st.plotly_chart(fig)

    df = pd.read_csv("data/Alfonso/ALFONSO 2020 - 2024.csv")

    df.columns = df.columns.str.strip()
   
    incident_counts = df['Address'].value_counts().reset_index()
    incident_counts.columns = ['Address', 'Total Incidents']

    # Create interactive horizontal bar chart
    fig = px.bar(
        incident_counts,
        x='Total Incidents',
        y='Address',
        orientation='h',
        title='Total Road Accidents per Address in Alfonso',
        labels={'Total Incidents': 'Number of Accidents'},
        hover_data={'Total Incidents': True, 'Address': True}
    )

    fig.update_layout(yaxis={'categoryorder': 'total ascending'}, height=800)

    st.plotly_chart(fig)


    df = pd.read_csv("data/Alfonso/ALFONSO 2020 - 2024.csv")
    
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df.dropna(subset=['Date'])


    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month_name()
    df['Month_Num'] = df['Date'].dt.month  # Useful for chronological sorting
    

# Now you can sort by year and month
    df = df.sort_values(['Year', 'Month_Num'])


    # Choose analysis type
    option = st.radio("Select View", ["Accidents Per Year", "Monthly Breakdown by Year", "Monthly Accidents (All Years)"])

    if option == "Accidents Per Year":
        yearly_counts = df['Year'].value_counts().sort_index()
        yearly_df = yearly_counts.reset_index()
        yearly_df.columns = ['Year', 'Total Accidents']

        fig = px.line(yearly_df, x='Year', y='Total Accidents',
                      title='Total Road Accidents Per Year', markers=True)
        fig.update_layout(xaxis=dict(tickmode='linear'))
        st.plotly_chart(fig)

    elif option == "Monthly Breakdown by Year":
        # Let user select a year
        selected_year = st.selectbox("Select Year", sorted(df['Year'].unique()))
        filtered_df = df[df['Year'] == selected_year]

        monthly_counts = filtered_df['Month'].value_counts().reindex([
            'January', 'February', 'March', 'April', 'May', 'June',
            'July', 'August', 'September', 'October', 'November', 'December'
        ]).fillna(0).astype(int)

        monthly_df = monthly_counts.reset_index()
        monthly_df.columns = ['Month', 'Total Accidents']

        fig = px.bar(monthly_df, x='Month', y='Total Accidents',
                     title=f'Total Accidents Per Month in {selected_year}')
        st.plotly_chart(fig)

    if option == "Accidents per Year":
        # Group by year
        yearly_counts = df['Year'].value_counts().sort_index()
        yearly_df = yearly_counts.reset_index()
        yearly_df.columns = ['Year', 'Total Accidents']

        fig_year = px.line(
            yearly_df,
            x='Year',
            y='Total Accidents',
            markers=True,
            title='Total Road Accidents per Year'
        )
        fig_year.update_layout(xaxis=dict(tickmode='linear'))
        st.plotly_chart(fig_year)

    elif option == "Accidents per Month (per Year)":
        # Let user choose year
        years = sorted(df['Year'].dropna().unique())
        selected_year = st.selectbox("Select year:", years)

        filtered = df[df['Year'] == selected_year]
        monthly_counts = (
            filtered.groupby(['Month', 'Month_Num'])
            .size()
            .reset_index(name='Total Accidents')
            .sort_values('Month_Num')
        )

        fig_month = px.bar(
            monthly_counts,
            x='Month',
            y='Total Accidents',
            title=f'Total Road Accidents per Month in {selected_year}'
        )
        st.plotly_chart(fig_month)

    elif option == "Monthly Accidents (All Years)":
    
        monthly_all_years = (
        df.groupby(['Month', 'Month_Num'])
        .size()
        .reset_index(name='Total Accidents')
        .sort_values('Month_Num')
    )

        fig = px.bar(
        monthly_all_years,
        x='Month',
        y='Total Accidents',
        title='Total Accidents Per Month (2020–2024)'
        )
        st.plotly_chart(fig)


    
    
elif page == "GMA":
    st.subheader("GMA Analysis")
   # st.image("data/qgis_maps/gma.png", caption="Hotspots in GMA")
   # st.image("data/qgis_maps/GMA/gma_heatmap.png", caption="Accident Heatmap in GMA")
   # st.image("data/qgis_maps/GMA/gma_month.png", caption="W.I.P")
   # st.image("data/qgis_maps/GMA/gma_year_1.png", caption="W.I.P")

    df = pd.read_csv("data/GMA/GMA 2020 - 2024.csv")

    # Count the number of accidents per year using the 'year' column directly
    yearly_counts = df['Year'].value_counts().sort_index()

# Convert to DataFrame for display or plotting
    yearly_df = yearly_counts.reset_index()
    yearly_df.columns = ['Year', 'Total Accidents']

# Plot using Plotly
    fig = px.line(
      yearly_df,
     x='Year',
     y='Total Accidents',
     title='Total Road Accidents per Year in GMA',
      markers=True
)

    fig.update_layout(xaxis=dict(tickmode='linear'))

# Show in Streamlit
    st.plotly_chart(fig)

    df = pd.read_csv("data/GMA/GMA 2020 - 2024.csv")

    df.columns = df.columns.str.strip()
   
    incident_counts = df['Address'].value_counts().reset_index()
    incident_counts.columns = ['Address', 'Total Incidents']

    # Create interactive horizontal bar chart
    fig = px.bar(
        incident_counts,
        x='Total Incidents',
        y='Address',
        orientation='h',
        title='Total Road Accidents per Barangay in GMA',
        labels={'Total Incidents': 'Number of Accidents'},
        hover_data={'Total Incidents': True, 'Address': True}
    )

    fig.update_layout(yaxis={'categoryorder': 'total ascending'}, height=800)

    st.plotly_chart(fig)


    df = pd.read_csv("data/GMA/GMA 2020 - 2024.csv")
    
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df.dropna(subset=['Date'])


    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month_name()
    df['Month_Num'] = df['Date'].dt.month  # Useful for chronological sorting
    

# Now you can sort by year and month
    df = df.sort_values(['Year', 'Month_Num'])



    # Choose analysis type
    option = st.radio("Select View", ["Accidents Per Year", "Monthly Breakdown by Year", "Monthly Accidents (All Years)"])

    if option == "Accidents Per Year":
        yearly_counts = df['Year'].value_counts().sort_index()
        yearly_df = yearly_counts.reset_index()
        yearly_df.columns = ['Year', 'Total Accidents']

        fig = px.line(yearly_df, x='Year', y='Total Accidents',
                      title='Total Road Accidents Per Year', markers=True)
        fig.update_layout(xaxis=dict(tickmode='linear'))
        st.plotly_chart(fig)

    elif option == "Monthly Breakdown by Year":
        # Let user select a year
        selected_year = st.selectbox("Select Year", sorted(df['Year'].unique()))
        filtered_df = df[df['Year'] == selected_year]

        monthly_counts = filtered_df['Month'].value_counts().reindex([
            'January', 'February', 'March', 'April', 'May', 'June',
            'July', 'August', 'September', 'October', 'November', 'December'
        ]).fillna(0).astype(int)

        monthly_df = monthly_counts.reset_index()
        monthly_df.columns = ['Month', 'Total Accidents']

        fig = px.bar(monthly_df, x='Month', y='Total Accidents',
                     title=f'Total Accidents Per Month in {selected_year}')
        st.plotly_chart(fig)

    if option == "Accidents per Year":
        # Group by year
        yearly_counts = df['Year'].value_counts().sort_index()
        yearly_df = yearly_counts.reset_index()
        yearly_df.columns = ['Year', 'Total Accidents']

        fig_year = px.line(
            yearly_df,
            x='Year',
            y='Total Accidents',
            markers=True,
            title='Total Road Accidents per Year'
        )
        fig_year.update_layout(xaxis=dict(tickmode='linear'))
        st.plotly_chart(fig_year)

    elif option == "Accidents per Month (per Year)":
        # Let user choose year
        years = sorted(df['Year'].dropna().unique())
        selected_year = st.selectbox("Select year:", years)

        filtered = df[df['Year'] == selected_year]
        monthly_counts = (
            filtered.groupby(['Month', 'Month_Num'])
            .size()
            .reset_index(name='Total Accidents')
            .sort_values('Month_Num')
        )

        fig_month = px.bar(
            monthly_counts,
            x='Month',
            y='Total Accidents',
            title=f'Total Road Accidents per Month in {selected_year}'
        )
        st.plotly_chart(fig_month)

    elif option == "Monthly Accidents (All Years)":
    
        monthly_all_years = (
        df.groupby(['Month', 'Month_Num'])
        .size()
        .reset_index(name='Total Accidents')
        .sort_values('Month_Num')
    )

        fig = px.bar(
        monthly_all_years,
        x='Month',
        y='Total Accidents',
        title='Total Accidents Per Month (2020–2024)'
        )
        st.plotly_chart(fig)

elif page == "Carmona":
    st.subheader("Carmona Analysis")
    #st.image("data/qgis_maps/carmona.png", caption="Hotspots in Carmona")

    df = pd.read_csv("data/Carmona/CARMONA 2020 - 2024.csv")

    

    # Count the number of accidents per year using the 'year' column directly
    yearly_counts = df['Year'].value_counts().sort_index()

# Convert to DataFrame for display or plotting
    yearly_df = yearly_counts.reset_index()
    yearly_df.columns = ['Year', 'Total Accidents']

# Plot using Plotly
    fig = px.line(
      yearly_df,
     x='Year',
     y='Total Accidents',
     title='Total Road Accidents per Year in Carmona',
      markers=True
)

    fig.update_layout(xaxis=dict(tickmode='linear'))

# Show in Streamlit
    st.plotly_chart(fig)

    df = pd.read_csv("data/Carmona/CARMONA 2020 - 2024.csv")

    df.columns = df.columns.str.strip()
   
    incident_counts = df['Address'].value_counts().reset_index()
    incident_counts.columns = ['Address', 'Total Incidents']

    # Create interactive horizontal bar chart
    fig = px.bar(
        incident_counts,
        x='Total Incidents',
        y='Address',
        orientation='h',
        title='Total Road Accidents per Barangay in Carmona',
        labels={'Total Incidents': 'Number of Accidents'},
        hover_data={'Total Incidents': True, 'Address': True}
    )

    fig.update_layout(yaxis={'categoryorder': 'total ascending'}, height=800)

    st.plotly_chart(fig)

    
    df = pd.read_csv("data/Carmona/CARMONA 2020 - 2024.csv")
    
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df.dropna(subset=['Date'])


    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month_name()
    df['Month_Num'] = df['Date'].dt.month  # Useful for chronological sorting
    

# Now you can sort by year and month
    df = df.sort_values(['Year', 'Month_Num'])


    # Choose analysis type
    option = st.radio("Select View", ["Accidents Per Year", "Monthly Breakdown by Year", "Monthly Accidents (All Years)"])

    if option == "Accidents Per Year":
        yearly_counts = df['Year'].value_counts().sort_index()
        yearly_df = yearly_counts.reset_index()
        yearly_df.columns = ['Year', 'Total Accidents']

        fig = px.line(yearly_df, x='Year', y='Total Accidents',
                      title='Total Road Accidents Per Year', markers=True)
        fig.update_layout(xaxis=dict(tickmode='linear'))
        st.plotly_chart(fig)

    elif option == "Monthly Breakdown by Year":
        # Let user select a year
        selected_year = st.selectbox("Select Year", sorted(df['Year'].unique()))
        filtered_df = df[df['Year'] == selected_year]

        monthly_counts = filtered_df['Month'].value_counts().reindex([
            'January', 'February', 'March', 'April', 'May', 'June',
            'July', 'August', 'September', 'October', 'November', 'December'
        ]).fillna(0).astype(int)

        monthly_df = monthly_counts.reset_index()
        monthly_df.columns = ['Month', 'Total Accidents']

        fig = px.bar(monthly_df, x='Month', y='Total Accidents',
                     title=f'Total Accidents Per Month in {selected_year}')
        st.plotly_chart(fig)

    if option == "Accidents per Year":
        # Group by year
        yearly_counts = df['Year'].value_counts().sort_index()
        yearly_df = yearly_counts.reset_index()
        yearly_df.columns = ['Year', 'Total Accidents']

        fig_year = px.line(
            yearly_df,
            x='Year',
            y='Total Accidents',
            markers=True,
            title='Total Road Accidents per Year'
        )
        fig_year.update_layout(xaxis=dict(tickmode='linear'))
        st.plotly_chart(fig_year)

    elif option == "Accidents per Month (per Year)":
        # Let user choose year
        years = sorted(df['Year'].dropna().unique())
        selected_year = st.selectbox("Select year:", years)

        filtered = df[df['Year'] == selected_year]
        monthly_counts = (
            filtered.groupby(['Month', 'Month_Num'])
            .size()
            .reset_index(name='Total Accidents')
            .sort_values('Month_Num')
        )

        fig_month = px.bar(
            monthly_counts,
            x='Month',
            y='Total Accidents',
            title=f'Total Road Accidents per Month in {selected_year}'
        )
        st.plotly_chart(fig_month)
        
    elif option == "Monthly Accidents (All Years)":
    
        monthly_all_years = (
        df.groupby(['Month', 'Month_Num'])
        .size()
        .reset_index(name='Total Accidents')
        .sort_values('Month_Num')
    )

        fig = px.bar(
        monthly_all_years,
        x='Month',
        y='Total Accidents',
        title='Total Accidents Per Month (2020–2024)'
        )
        st.plotly_chart(fig)
# Sidebar – Controls
st.sidebar.title("Display Options")
show_clustering = st.sidebar.checkbox("Enable Interactive Clustering")

if show_clustering:
    method = st.sidebar.radio("Select Clustering Method", ["DBSCAN", "KMeans"])
    if method == "KMeans":
        k = st.sidebar.slider("Number of Clusters", min_value=1, max_value=10, value=3)
    
    if st.button("Run Clustering"):
        # Load data and define coords
        #df = pd.read_csv("data/Alfonso/ALFONSO 2020 - 2024.csv")
       # df = pd.read_csv("data/Carmona/CARMONA 2020 - 2024.csv")
        df = pd.read_csv("data/GMA/GMA 2020 - 2024.csv")
        coords = df[['Latitude', 'Longitude']].values

        if method == "DBSCAN":
            st.subheader("DBSCAN Clustering")
            coords_rad = np.radians(coords)
            eps_rad = 0.3 / 6371.0  # 300 meters
            db = DBSCAN(eps=eps_rad, min_samples=4, metric='haversine')
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
