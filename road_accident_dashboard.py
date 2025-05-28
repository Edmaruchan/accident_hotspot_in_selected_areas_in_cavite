import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
from PIL import Image

st.set_page_config(
    page_title="Your App Title",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="auto",
    menu_items={
        'Get Help': None,
        'Report a bug': None,
        'About': None,
    }
)

hide_streamlit_style = """
    <style>
    header[data-testid="stHeader"] {
        display: none;
    }
    footer {
        visibility: hidden;
    }
    </style>
"""

st.markdown(hide_streamlit_style, unsafe_allow_html=True)


st.title("Road Accident Analysis in Selected Areas in Cavite")

st.subheader("Welcome to the Road Accident Dashboard")
page = st.sidebar.radio("Select Area", ["Overview", "Alfonso", "GMA", "Carmona"])

# Pages
if page == "Overview":
 
  df = pd.read_csv("data/ALL.csv")

  # Count the number of accidents per year using the 'year' column directly
  yearly_counts = df['Year'].value_counts().sort_index()

# Convert to DataFrame for display or plotting
  yearly_df = yearly_counts.reset_index()
  yearly_df.columns = ['Year', 'Total Accidents']

  st.title("Total Road Accidents (2020–2024)")
  
# Plot using Plotly
  fig = px.line(
     yearly_df,
     x='Year',
     y='Total Accidents',
     markers=True
)

  fig.update_layout(
    xaxis=dict(tickmode='linear'),
    dragmode=False
    )

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

  st.title("Total Road Accidents per Barangay (Alfonso, Carmona, GMA) in Cavite")

# Plot interactive horizontal bar chart
  fig = px.bar(
    incident_counts,
    x='Total Incidents',
    y='Address',
    orientation='h',
    labels={'Total Incidents': 'Number of Accidents'},
    hover_data={'Total Incidents': True, 'Address': True}
  )

  fig.update_layout(
    dragmode=False,
    yaxis={'categoryorder': 'total ascending'},
    height=1000,         # Adjust height here
    width=900,           # Add custom width
    margin=dict(l=150),  # More left margin for long address labels
  )

  st.plotly_chart(fig, use_container_width=False)
        
################ YEARLY DISTRIBUTION OF ACCIDENTS USING PLOTLY ##################        
  
  st.title("Yearly Accident Trends in 3 Municipalities(2020–2024)")
  
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
            template="plotly_white",
            dragmode=False,
        )

        # Show Plotly figure in Streamlit
  st.plotly_chart(fig, use_container_width=True)
        
        
################## MONTHLY DISTRIBUTION OF ACCIDENTS ##################

  st.title("Monthly Accident Trends Per Year")

        # File paths (you can replace these or use a dropdown)
  municipality_options = {
            "GMA": "data/GMA/GMA 2020 - 2024.csv",
            "Alfonso": "data/Alfonso/ALFONSO 2020 - 2024.csv",
            "Carmona": "data/Carmona/CARMONA 2020 - 2024.csv"
        }

  selected_municipality = st.radio("Select Municipality", list(municipality_options.keys()), key="municipality_selection1")
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
            title=f"Monthly Accident Trends by Year:  {selected_municipality}",
            xaxis_title="Year",
            yaxis_title="Number of Accidents",
            legend_title="Month",
            template="plotly_white",
            dragmode=False
        )

  st.plotly_chart(fig, use_container_width=True)
  
  
  ############# SEASONAL DISTRIBUTION OF ACCIDENTS ############
  
  df = pd.read_csv("data/seasons all.csv")
    
  season_order = ['Cool Dry Season', 'Hot Dry Season', 'Wet Season']
  df['Season'] = pd.Categorical(df['Season'], categories=season_order, ordered=True)

# Plot line chart
  fig = px.line(
    df,
    x='Season',
    y='Number of Accidents',
    color='Municipality',
    markers=True,
    title='Seasonal Road Accidents per Municipality'
)

  st.plotly_chart(fig)
  
  
  ############ WEEKLY DISTRIBUTION OF ACCIDENTS ############
  
  st.title("Weekly Accident Trends by Year")
  
  municipality_options = {
    "Alfonso": "data/Alfonso/ALFONSO weekly 2020 - 2024.csv",
    "GMA": "data/GMA/GMA weekly 2020 - 2024.csv",
    "Carmona": "data/Carmona/CARMONA weekly 2020 - 2024.csv"
    
}

# Radio button for municipality selection
  selected_municipality = st.radio("Select Municipality", list(municipality_options.keys()), key="municipality_selection2")

# Load corresponding CSV
  path = municipality_options[selected_municipality]
  df = pd.read_csv(path)

# Ensure proper order of days
  days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
  df['Day'] = pd.Categorical(df['Day'], categories=days_order, ordered=True)

# Plot line chart
  fig = px.line(
    df,
    x='Year',
    y='Total Accidents',
    color='Day',
    markers=True,
    title=f'Weekly Accident Trends per Year:  {selected_municipality}',
    labels={'Total Accidents': 'Total Accidents', 'Year': 'Year'}
)

  st.plotly_chart(fig, use_container_width=True, config={"scrollZoom": False,})
  
  
  
############# TIME OF DAY DISTRIBUTION OF ACCIDENTS ############

  municipality_options = {
    "Alfonso": "data/Alfonso/ALFONSO 2020 - 2024.csv",
    "GMA": "data/GMA/GMA 2020 - 2024.csv",
    "Carmona": "data/Carmona/CARMONA 2020 - 2024.csv"
}
  
  df.columns = df.columns.str.strip()


# --- Select Municipality ---
  selected_municipality = st.radio("Select Municipality", list(municipality_options.keys()), key="municipality_selection3")
  file_path = municipality_options[selected_municipality]

# --- Load Data ---
  df = pd.read_csv(file_path)
  df.columns = df.columns.str.strip()  # removes leading/trailing spaces from column names
  
# --- Preprocess ---
  df['Time'] = pd.to_datetime(df['Time'], format='%H:%M:%S', errors='coerce').dt.time  # Ensure time parsed correctly

  def categorize_time(t):
    if pd.isna(t):
        return None
    if t >= pd.to_datetime('00:00').time() and t <= pd.to_datetime('05:59').time():
        return 'EARLY MORNING'
    elif t >= pd.to_datetime('06:00').time() and t <= pd.to_datetime('11:59').time():
        return 'MORNING'
    elif t >= pd.to_datetime('12:00').time() and t <= pd.to_datetime('17:59').time():
        return 'AFTERNOON'
    elif t >= pd.to_datetime('18:00').time() and t <= pd.to_datetime('23:59').time():
        return 'EVENING'

  df['Time of Day'] = df['Time'].apply(categorize_time)

# --- Group and Prepare for Plotting ---
  time_order = ['EARLY MORNING', 'MORNING', 'NOON', 'AFTERNOON', 'EVENING']
  grouped = df.groupby(['Year', 'Time of Day']).size().reset_index(name='Total Accidents')
  grouped['Time of Day'] = pd.Categorical(grouped['Time of Day'], categories=time_order, ordered=True)

# Define a color map for each time of day
  time_color_map = {
      'EARLY MORNING': "#000CAD",  # blue
      'MORNING': "#00C0DA",        # green
      'NOON': "#FFD085",           # orange
      'AFTERNOON': "#FF9100",      # purple
      'EVENING': "#050029"         # red
  }

  # --- Plot ---
  fig = px.line(
      grouped,
      x='Year',
      y='Total Accidents',
      color='Time of Day',
      color_discrete_map=time_color_map,  # Apply custom colors
      markers=True,
      title=f'Time of Day Accident Trends by Year: {selected_municipality}'
  )

  st.title("Time of Day Accident Trends in 3 Municipalities (2020–2024)")
  st.plotly_chart(fig, use_container_width=True, config={"scrollZoom": False})

############# Weekend vs Weekday Distribution of Accidents ############
  # Reshape the DataFrame from wide to long format
  st.title("Weekdays and Weekends trend in 3 Municipality (2020–2024)")
  
  df = pd.read_csv("data/weekends vs weekdays all.csv")
  df_long = pd.melt(df, id_vars='Municipality', var_name='Day Type', value_name='Number of Accidents')

  # Optional: Set order of Day Type
  df_long['Day Type'] = pd.Categorical(df_long['Day Type'], categories=['Weekdays', 'Weekends'], ordered=True)

  # Plot using Plotly
  fig = px.line(
      df_long,
      x='Day Type',
      y='Number of Accidents',
      color='Municipality',
      markers=True,
      title='Accidents on Weekdays vs Weekends by Municipality (2020–2024)',
  )

  st.plotly_chart(fig, config={"scrollZoom": False,})
  
############# ALFONSO ############

  
elif page == "Alfonso":
    st.subheader("Alfonso Analysis")
    
    
    
    image_folder = "data/qgis_maps/Alfonso"
    image_files = sorted([f for f in os.listdir(image_folder) if f.endswith((".jpg", ".png", ".jpeg"))])

    # --- Display image
    current_image = Image.open(os.path.join(image_folder, image_files[st.session_state.index]))
    st.image(current_image, use_container_width=True, caption=image_files[st.session_state.index])
    df = pd.read_csv("data/Alfonso/ALFONSO 2020 - 2024.csv")
    
    # --- Session state to keep track of current image
    if "index" not in st.session_state:
        st.session_state.index = 0

    # --- Navigation buttons
    col1, col_spacer, col2 = st.columns([1, 10, 1])  # Adjust column widths as needed
    with col1:
      with st.container():
        if st.button("⬅️"):
            st.session_state.index = (st.session_state.index - 1) % len(image_files)
    with col2:
      with st.container():
        if st.button("➡️"):
            st.session_state.index = (st.session_state.index + 1) % len(image_files)

    st.write("")
    
    

    
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df.dropna(subset=['Date'])


    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month_name()
    df['Month_Num'] = df['Date'].dt.month  # Useful for chronological sorting
    

#  Now you can sort by year and month
    df = df.sort_values(['Year', 'Month_Num'])


    # Choose analysis type
    option = st.radio("Select one", ["Accidents Per Year", "Monthly Breakdown by Year", "Monthly Accidents (All Years)"])

    if option == "Accidents Per Year":

        st.subheader("Total Accidents Per Year in Alfonso")
        yearly_counts = df['Year'].value_counts().sort_index()
        yearly_df = yearly_counts.reset_index()
        yearly_df.columns = ['Year', 'Total Accidents']

        fig = px.line(yearly_df, x='Year', y='Total Accidents',
                      title='Total Road Accidents Per Year', markers=True)
        fig.update_layout(xaxis=dict(tickmode='linear'), dragmode=False,)
        st.plotly_chart(fig)

    elif option == "Monthly Breakdown by Year":
        st.subheader("Total Accidents Per Month in Alfonso")
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
        fig_year.update_layout(xaxis=dict(tickmode='linear'), dragmode=False,)
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

        st.subheader("Total Accidents Per Month (2020–2024) in Alfonso")
        monthly_all_years = (
        df.groupby(['Month', 'Month_Num'])
        .size()
        .reset_index(name='Total Accidents')
        .sort_values('Month_Num')
    )

        fig = px.bar(
        monthly_all_years,
        x='Month',
        y='Total Accidents'
        )
        st.plotly_chart(fig)

########## ALL ACCIDENTS PER ADDRESS IN ALFONSO ##########

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

    fig.update_layout(yaxis={'categoryorder': 'total ascending'}, 
                      height=800,
                      dragmode=False
                      )

    st.plotly_chart(fig)
    
    
elif page == "GMA":
    st.subheader("GMA Analysis")

    image_folder = "data/qgis_maps/GMA"
    image_files = sorted([f for f in os.listdir(image_folder) if f.endswith((".jpg", ".png", ".jpeg"))])

    # --- Display image
    current_image = Image.open(os.path.join(image_folder, image_files[st.session_state.index]))
    st.image(current_image, use_container_width=True, caption=image_files[st.session_state.index])

    
    # --- Session state to keep track of current image
    if "index" not in st.session_state:
        st.session_state.index = 0

    # --- Navigation buttons
    col1, col_spacer, col2 = st.columns([1, 10, 1])  # Adjust column widths as needed
    with col1:
      with st.container():
        if st.button("⬅️"):
            st.session_state.index = (st.session_state.index - 1) % len(image_files)
    with col2:
      with st.container():
        if st.button("➡️"):
            st.session_state.index = (st.session_state.index + 1) % len(image_files)  

    df = pd.read_csv("data/GMA/GMA 2020 - 2024.csv")
    
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df.dropna(subset=['Date'])


    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month_name()
    df['Month_Num'] = df['Date'].dt.month  # Useful for chronological sorting
    

#  Now you can sort by year and month
    df = df.sort_values(['Year', 'Month_Num'])


    # Choose analysis type
    option = st.radio("Select View", ["Accidents Per Year", "Monthly Breakdown by Year", "Monthly Accidents (All Years)"])

    if option == "Accidents Per Year":

        st.subheader("Total Accidents Per Year in GMA")
        yearly_counts = df['Year'].value_counts().sort_index()
        yearly_df = yearly_counts.reset_index()
        yearly_df.columns = ['Year', 'Total Accidents']

        fig = px.line(yearly_df, x='Year', y='Total Accidents',
                      title='Total Road Accidents Per Year', markers=True)
        fig.update_layout(xaxis=dict(tickmode='linear'), dragmode=False,)
        st.plotly_chart(fig)

    elif option == "Monthly Breakdown by Year":
        st.subheader("Total Accidents Per Month in GMA")
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
        fig_year.update_layout(xaxis=dict(tickmode='linear'), dragmode=False,)
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

        st.subheader("Total Accidents Per Month (2020–2024) in GMA")
        monthly_all_years = (
        df.groupby(['Month', 'Month_Num'])
        .size()
        .reset_index(name='Total Accidents')
        .sort_values('Month_Num')
    )

        fig = px.bar(
        monthly_all_years,
        x='Month',
        y='Total Accidents'
        )
        st.plotly_chart(fig)
        

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

    fig.update_layout(yaxis={'categoryorder': 'total ascending'}, height=800, dragmode=False)

    st.plotly_chart(fig)



        
elif page == "Carmona":
    st.subheader("Carmona Analysis")
    
    image_folder = "data/qgis_maps/Carmona"
    image_files = sorted([f for f in os.listdir(image_folder) if f.endswith((".jpg", ".png", ".jpeg"))])

    # --- Display image
    current_image = Image.open(os.path.join(image_folder, image_files[st.session_state.index]))
    st.image(current_image, use_container_width=True, caption=image_files[st.session_state.index])
  
    
    # --- Session state to keep track of current image
    if "index" not in st.session_state:
        st.session_state.index = 0

    # --- Navigation buttons
    col1, col_spacer, col2 = st.columns([1, 10, 1])  # Adjust column widths as needed
    with col1:
      with st.container():
        if st.button("⬅️"):
            st.session_state.index = (st.session_state.index - 1) % len(image_files)
    with col2:
      with st.container():
        if st.button("➡️"):
            st.session_state.index = (st.session_state.index + 1) % len(image_files)

    df = pd.read_csv("data/Carmona/CARMONA 2020 - 2024.csv")
    
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df.dropna(subset=['Date'])


    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month_name()
    df['Month_Num'] = df['Date'].dt.month  # Useful for chronological sorting
    

#  Now you can sort by year and month
    df = df.sort_values(['Year', 'Month_Num'])


    # Choose analysis type
    option = st.radio("Select One", ["Accidents Per Year", "Monthly Breakdown by Year", "Monthly Accidents (All Years)"])

    if option == "Accidents Per Year":

        st.subheader("Total Accidents Per Year in Carmona")
        yearly_counts = df['Year'].value_counts().sort_index()
        yearly_df = yearly_counts.reset_index()
        yearly_df.columns = ['Year', 'Total Accidents']

        fig = px.line(yearly_df, x='Year', y='Total Accidents',
                      title='Total Road Accidents Per Year', markers=True)
        fig.update_layout(xaxis=dict(tickmode='linear'), dragmode=False,)
        st.plotly_chart(fig)

    elif option == "Monthly Breakdown by Year":
        st.subheader("Total Accidents Per Month in Carmona")
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
        fig_year.update_layout(xaxis=dict(tickmode='linear'), dragmode=False,)
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

        st.subheader("Total Accidents Per Month (2020–2024) in Carmona")
        monthly_all_years = (
        df.groupby(['Month', 'Month_Num'])
        .size()
        .reset_index(name='Total Accidents')
        .sort_values('Month_Num')
    )

        fig = px.bar(
        monthly_all_years,
        x='Month',
        y='Total Accidents'
        )
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

    fig.update_layout(yaxis={'categoryorder': 'total ascending'}, height=800, dragmode=False)

    st.plotly_chart(fig)

    
    
            
            