import pandas as pd

# Load the dataset
data = pd.read_csv('cleaned_dataset.csv')

# Display the first few rows of the dataset
print(data.head())

from sklearn.preprocessing import StandardScaler

# Assuming 'features' contains the column names of the features
features = data.columns

# Standardize the data
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data[features])

from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

# Perform K-Means Clustering
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(data_scaled)

# Add the cluster labels to the original data
data['Cluster'] = clusters

# Visualize the clusters using a scatter plot
plt.figure(figsize=(10, 6))
sns.scatterplot(x=data_scaled[:, 0], y=data_scaled[:, 1], hue=clusters, palette='viridis')
plt.title('K-Means Clustering')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()

from sklearn.metrics import silhouette_score

# Calculate the Silhouette Score
silhouette_avg = silhouette_score(data_scaled, clusters)
print(f'Silhouette Score: {silhouette_avg}')

from sklearn.cluster import DBSCAN

# Perform DBSCAN Clustering
dbscan = DBSCAN(eps=0.5, min_samples=5)
dbscan_clusters = dbscan.fit_predict(data_scaled)

# Calculate the Silhouette Score for DBSCAN
dbscan_silhouette_avg = silhouette_score(data_scaled, dbscan_clusters)
print(f'DBSCAN Silhouette Score: {dbscan_silhouette_avg}')

# Visualize the DBSCAN clusters
plt.figure(figsize=(10, 6))
sns.scatterplot(x=data_scaled[:, 0], y=data_scaled[:, 1], hue=dbscan_clusters, palette='viridis')
plt.title('DBSCAN Clustering')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()
