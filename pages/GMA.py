import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

import streamlit as st
from PIL import Image


st.subheader("GMA Analysis")


df = pd.read_csv("data/GMA/GMA 2020 - 2024.csv")

# Convert date column
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')  # errors='coerce' handles invalid dates
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month_name()

# Sidebar controls for filtering
st.sidebar.subheader("Statistics Options")
stats_view = st.sidebar.selectbox("View Statistics By", ["Year", "Month"])

# Show statistics based on selection
if stats_view == "Year":
    st.subheader("Total Incidents per Year")
    yearly_counts = df['Year'].value_counts().sort_index()
    st.bar_chart(yearly_counts)
    st.write("The year 2022 has the highest number of incidents.")
    st.write("The year 2020 has the lowest number of incidents.")

elif stats_view == "Month":
    st.subheader("Total Incidents per Month")
    month_order = ["January", "February", "March", "April", "May", "June",
                   "July", "August", "September", "October", "November", "December"]
    monthly_counts = df['Month'].value_counts().reindex(month_order)
    st.bar_chart(monthly_counts)
    st.write("June and September have the highest number of incidents.")
    st.write("The months of April and May have the lowest number of incidents.")