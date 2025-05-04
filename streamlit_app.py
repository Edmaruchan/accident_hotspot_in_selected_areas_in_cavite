import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt

# Example: CSV with 'latitude' and 'longitude' columns
df = pd.read_csv("/accident_hotspot_in_selected_areas_in_cavite/data/gdp_data.csv")

# Convert to radians for haversine
coords = np.radians(df[['Latitude', 'Longitude']].values)

# Define epsilon in radians (e.g., 300 meters ≈ 0.003 rad)
# Earth radius ≈ 6,371 km → 300 meters = 0.3 km

eps_rad = 0.3 / 6371.0
min_samples = 3

"""for e in [0.2, 0.3, 0.5]:
    for m in [2, 3, 5]:
        db = DBSCAN(eps=e/6371.0, min_samples=m, metric='haversine')
        labels = db.fit_predict(coords)
        print(f"eps={e*1000:.0f}m, min_samples={m}, clusters={len(set(labels)) - (1 if -1 in labels else 0)}")"""



# Run DBSCAN
db = DBSCAN(eps=eps_rad, min_samples=3, metric='haversine')
df['cluster'] = db.fit_predict(coords)

# Check cluster counts
print(df['cluster'].value_counts())

# Optional: Plot results
plt.figure(figsize=(8, 6))
plt.scatter(df['Longitude'], df['Latitude'], c=df['cluster'], cmap='tab20', s=10)
plt.title("DBSCAN Clustering of Road Accidents")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.grid(True)
plt.show()