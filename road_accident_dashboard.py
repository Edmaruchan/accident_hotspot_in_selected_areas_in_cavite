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
        
################ YEARLY DISTRIBUTION OF ACCIDENTS USING PLOTLY ##################        

  gma_path = "data/GMA/GMA 2020 - 2024.csv"
  alfonso_path = "data/Alfonso/ALFONSO 2020 - 2024.csv"
  carmona_path = "data/Carmona/CARMONA 2020 - 2024.csv"

     # Function to load and process each file
  def load_yearly_counts(path):
      
    df = pd.read_csv(path, parse_dates=['Date'])
    df['Year'] = df['Date'].dt.year
    return df.groupby('Year').size().reset_index(name='accidents')

        # Load and process
  gma_df = load_yearly_counts(gma_path)
  alfonso_df = load_yearly_counts(alfonso_path)
  carmona_df = load_yearly_counts(carmona_path)

        # Plot using Plotly
  fig = go.Figure()

  fig.add_trace(go.Scatter(x=gma_df['Year'], y=gma_df['accidents'],
    mode='lines+markers', name='GMA', line=dict(color='red')))

  fig.add_trace(go.Scatter(x=alfonso_df['Year'], y=alfonso_df['accidents'],
    mode='lines+markers', name='Alfonso', line=dict(color='green')))

  fig.add_trace(go.Scatter(x=carmona_df['Year'], y=carmona_df['accidents'],
                                mode='lines+markers', name='Carmona', line=dict(color='blue')))

        # Customize layout
  fig.update_layout(
            title="Yearly Road Accident Distribution (2020–2024)",
            xaxis_title="Year",
            yaxis_title="Number of Accidents",
            legend_title="Municipality",
            template="plotly_white"
        )

        # Show Plotly figure in Streamlit
  st.plotly_chart(fig, use_container_width=True)
        
        
################## MONTHLY DISTRIBUTION OF ACCIDENTS ##################

  st.title("📆 Monthly Accident Trends Over Years (One Municipality)")

        # File paths (you can replace these or use a dropdown)
  municipality_options = {
            "GMA": "data/GMA/GMA 2020 - 2024.csv",
            "Alfonso": "data/Alfonso/ALFONSO 2020 - 2024.csv",
            "Carmona": "data/Carmona/CARMONA 2020 - 2024.csv"
        }

  selected_municipality = st.selectbox("Select Municipality", list(municipality_options.keys()))
  path = municipality_options[selected_municipality]

        # Load and process data
  df = pd.read_csv(path, parse_dates=['Date'])
  df['year'] = df['Date'].dt.year
  df['month_name'] = df['Date'].dt.strftime('%B')
  df['month_num'] = df['Date'].dt.month

        # Group by year and month
  monthly_trend = df.groupby(['year', 'month_num', 'month_name']).size().reset_index(name='accidents')
  monthly_trend.sort_values(by=['month_num', 'year'], inplace=True)

        # Plot
  fig = go.Figure()

        # Loop through each month to create a line
  for month_num in range(1, 13):
            month_df = monthly_trend[monthly_trend['month_num'] == month_num]
            fig.add_trace(go.Scatter(
                x=month_df['year'],
                y=month_df['accidents'],
                mode='lines+markers',
                name=month_df['month_name'].iloc[0]
            ))

        # Layout
  fig.update_layout(
            title=f"Monthly Accident Trends by Year ({selected_municipality})",
            xaxis_title="Year",
            yaxis_title="Number of Accidents",
            legend_title="Month",
            template="plotly_white"
        )

  st.plotly_chart(fig, use_container_width=True)
############# ALFONSO ############

  
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
            
            
            