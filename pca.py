import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# --------------------------
# LOAD IRIS DATASET
# --------------------------
iris = load_iris()
iris_df = pd.DataFrame(
    data=np.c_[iris['data'], iris['target']],
    columns=iris['feature_names'] + ['target']
)

# Features and target
X = iris_df.drop('target', axis=1)
y = iris_df['target']

# --------------------------
# FEATURE SCALING
# --------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --------------------------
# APPLY PCA
# --------------------------
pca = PCA()
X_pca = pca.fit_transform(X_scaled)

# Explained variance ratio
explained_variance_ratio = pca.explained_variance_ratio_

# Plot cumulative explained variance
plt.figure(figsize=(8,6))
plt.plot(np.cumsum(explained_variance_ratio), marker='o', linestyle='--', color='blue')
plt.title('Cumulative Explained Variance by PCA Components')
plt.xlabel('Number of Principal Components')
plt.ylabel('Cumulative Explained Variance')
plt.grid(True)
plt.show()

# Determine number of components to explain 95% variance
cumulative_variance_ratio = np.cumsum(explained_variance_ratio)
n_components = np.argmax(cumulative_variance_ratio >= 0.95) + 1
print(f"Number of principal components to explain 95% variance: {n_components}")

# Apply PCA with selected number of components
pca = PCA(n_components=n_components)
X_reduced = pca.fit_transform(X_scaled)

# --------------------------
# VISUALIZE REDUCED-DIMENSION DATA
# --------------------------
plt.figure(figsize=(9,6))
scatter = plt.scatter(
    X_reduced[:, 0], X_reduced[:, 1],
    c=y, cmap='viridis', s=50, alpha=0.7
)
plt.title('Iris Dataset in Reduced-Dimensional Space (PCA)')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.colorbar(scatter, label='Target Class')
plt.grid(True)
plt.show()
