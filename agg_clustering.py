import random
import numpy as np
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt

# Set the seed for reproducibility
np.random.seed(42)

# Function to generate random latitude and longitude pairs
def generate_random_lat_lon_pairs(num_points):
    latitudes = np.random.uniform(low=-90.0, high=90.0, size=num_points)
    longitudes = np.random.uniform(low=-180.0, high=180.0, size=num_points)
    return list(zip(latitudes, longitudes))

# Generate 1000 random latitude-longitude pairs
num_points = 1000
lat_lon_pairs = generate_random_lat_lon_pairs(num_points)

# Generate unique customer codes
customer_codes = [f'C{str(i).zfill(4)}' for i in range(1, num_points + 1)]

# Create a dictionary to store customer codes and corresponding lat-lon pairs
customer_data = dict(zip(customer_codes, lat_lon_pairs))

# Convert lat-lon pairs to numpy array
X = np.array(lat_lon_pairs)

# Apply Agglomerative Clustering
cluster_count = 5  # You can adjust this based on your requirements
agg_cluster = AgglomerativeClustering(n_clusters=cluster_count, affinity='euclidean', linkage='ward')
agg_labels = agg_cluster.fit_predict(X)

# Plot the clusters
plt.scatter(X[:, 1], X[:, 0], c=agg_labels, cmap='viridis', s=10)
plt.title('Agglomerative Clustering of Lat-Lon Pairs')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.show()
