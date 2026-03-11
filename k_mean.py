import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler

# --------------------------
# GENERATE SYNTHETIC DATA
# --------------------------
X, _ = make_blobs(
    n_samples=300, 
    centers=4, 
    cluster_std=1.0, 
    random_state=42
)

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --------------------------
# ELBOW METHOD TO FIND OPTIMAL K
# --------------------------
wcss = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

# Print WCSS values
print("\nWCSS Values for Different K:")
for i, val in enumerate(wcss, start=1):
    print(f"K={i}: WCSS={val:.2f}")

# Plot the Elbow curve
plt.figure(figsize=(8, 5))
plt.plot(range(1, 11), wcss, marker='o', linestyle='--', color='blue')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('WCSS')
plt.title('Elbow Method for Optimal K')
plt.grid(True)
plt.show()

# --------------------------
# APPLY K-MEANS WITH OPTIMAL K
# --------------------------
optimal_k = 4
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
y_kmeans = kmeans.fit_predict(X_scaled)

# --------------------------
# PRINT CLUSTER INFORMATION
# --------------------------
print("\nCluster Assignments for First 10 Points:")
for i in range(10):
    print(f"Point {i+1}: Cluster {y_kmeans[i]}")

print("\nCluster Centroids:")
print(kmeans.cluster_centers_)

# --------------------------
# VISUALIZE CLUSTERS
# --------------------------
plt.figure(figsize=(8, 6))
plt.scatter(
    X_scaled[:, 0], X_scaled[:, 1],
    c=y_kmeans, cmap='viridis', edgecolors='k', s=50, label='Data Points'
)
plt.scatter(
    kmeans.cluster_centers_[:, 0],
    kmeans.cluster_centers_[:, 1],
    s=200,
    c='red',
    marker='X',
    label='Centroids'
)
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('K-Means Clustering Results')
plt.legend()
plt.grid(True)
plt.show()
