import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datashader as ds
import datashader.transfer_functions as tf
from datashader.mpl_ext import dsshow
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from geopy.distance import geodesic
from scipy.spatial import KDTree
from sklearn.cluster import KMeans

# Load data
df = pd.read_csv('data/nb.csv')


# Basic statistics
print("Basic Statistics:\n", df.describe())

# Missing values
print("Missing Values:\n", df.isnull().sum())

# Treat missing values (example: fill with median)
df.fillna(df.median(), inplace=True)

# Outlier detection (example: using Z-score)
z_scores = np.abs((df - df.mean()) / df.std())
df = df[(z_scores < 3).all(axis=1)]

# Basic visualizations
plt.figure(figsize=(10, 6))
sns.histplot(df['order_location_latitude'], kde=True)
plt.title('Distribution of Order Location Latitude')
plt.show()

plt.figure(figsize=(10, 6))
sns.histplot(df['order_location_longitude'], kde=True)
plt.title('Distribution of Order Location Longitude')
plt.show()

# Geospatial Visualization using Datashader
canvas = ds.Canvas(plot_width=400, plot_height=400)
agg = canvas.points(df, 'order_location_longitude', 'order_location_latitude')
img = tf.shade(agg)
tf.set_background(img, "black").to_pil().show()

# More detailed visualizations and insights
# Correlation heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()

# Save cleaned data
df.to_csv('data/cleaned_geospatial_data.csv', index=False)

# Additional feature creation based on time and location
df['start_time'] = pd.to_datetime(df['start_time'])
df['end_time'] = pd.to_datetime(df['end_time'])

# Extracting features from datetime
df['hour_of_day'] = df['start_time'].dt.hour
df['day_of_week'] = df['start_time'].dt.dayofweek
df['is_weekend'] = df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)

# Placeholder for external data features (rain, holiday, traffic)
df['rain'] = np.random.choice([0, 1], size=len(df))  # Placeholder
df['holiday'] = np.random.choice([0, 1], size=len(df))  # Placeholder
df['traffic_condition'] = np.random.choice(['low', 'medium', 'high'], size=len(df))  # Placeholder

# One-hot encode categorical features
df = pd.get_dummies(df, columns=['traffic_condition'], drop_first=True)

# Feature extraction example - travel time
df['travel_time'] = (df['end_time'] - df['start_time']).dt.total_seconds() / 60  # in minutes

print("Head of dataframe with new features:\n", df.head())

# Save the dataframe with new features
df.to_csv('data/enriched_geospatial_data.csv', index=False)

# Feature scaling
scaler = StandardScaler()
df_scaled = df.copy()
df_scaled[['order_location_latitude', 'order_location_longitude', 'travel_time']] = scaler.fit_transform(df_scaled[['order_location_latitude', 'order_location_longitude', 'travel_time']])

print("Head of scaled dataframe:\n", df_scaled.head())

# Save the scaled dataframe
df_scaled.to_csv('data/scaled_geospatial_data.csv', index=False)

def compute_distance(row):
    start_coords = (row['driver_location_latitude'], row['driver_location_longitude'])
    end_coords = (row['order_destination_latitude'], row['order_destination_longitude'])
    return geodesic(start_coords, end_coords).meters

df['distance_m'] = df.apply(compute_distance, axis=1)

# Compute driving speed (placeholder example)
df['driving_speed'] = df['distance_m'] / df['travel_time']  # in meters per minute

print("Head of dataframe with distance and speed:\n", df.head())

# Save the dataframe with distance and speed
df.to_csv('data/feature_enriched_geospatial_data.csv', index=False)

# Create KDTree for efficient radius searches
locations = df[['order_location_latitude', 'order_location_longitude']].values
kdtree = KDTree(locations)

def count_neighbors_within_radius(row, radius=500):
    point = (row['order_location_latitude'], row['order_location_longitude'])
    indices = kdtree.query_ball_point(point, radius / 1000)  # radius in km
    return len(indices)

df['riders_within_500m'] = df.apply(count_neighbors_within_radius, axis=1)

print("Head of dataframe with riders within 500m:\n", df.head())

# Save the dataframe with riders count
df.to_csv('data/riders_enriched_geospatial_data.csv', index=False)

# Clustering starting locations
kmeans_start = KMeans(n_clusters=5)
df['start_location_cluster'] = kmeans_start.fit_predict(df[['driver_location_latitude', 'driver_location_longitude']])

# Clustering destination locations
kmeans_dest = KMeans(n_clusters=5)
df['destination_location_cluster'] = kmeans_dest.fit_predict(df[['order_destination_latitude', 'order_destination_longitude']])

print("Head of dataframe with clusters:\n", df.head())

# Save the dataframe with clusters
df.to_csv('data/clustered_geospatial_data.csv', index=False)

# Purpose-driven visualization using Datashader
canvas = ds.Canvas(plot_width=800, plot_height=800)
agg = canvas.points(df, 'order_location_longitude', 'order_location_latitude')
img = tf.shade(agg)
tf.set_background(img, "black").to_pil().show()

# Cluster visualization
cluster_canvas = ds.Canvas(plot_width=800, plot_height=800)
agg = cluster_canvas.points(df, 'order_location_longitude', 'order_location_latitude', ds.count_cat('start_location_cluster'))
img = tf.shade(agg)
tf.set_background(img, "black").to_pil().show()
