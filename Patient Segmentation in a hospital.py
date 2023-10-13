#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# Load a sample patient dataset (replace with your actual dataset)
data = pd.read_csv('sample_patient_data.csv')

# Assuming you have features for segmentation, let's say 'Age' and 'Health Score'
# You can use additional features depending on your dataset
X = data[['Age', 'Health Score']]

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply PCA to reduce dimensionality
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Determine the optimal number of clusters using the Elbow Method
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(X_pca)
    wcss.append(kmeans.inertia_)

# Plot the Elbow Method results to find the optimal number of clusters
plt.figure(figsize=(8, 6))
plt.plot(range(1, 11), wcss, marker='o', linestyle='--')
plt.title('Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.show()

# Based on the Elbow Method, let's choose the optimal number of clusters
kmeans = KMeans(n_clusters=3, init='k-means++', random_state=42)
kmeans.fit(X_pca)

# Add the cluster labels to the original dataset
data['Cluster'] = kmeans.labels_

# Visualize the segmented patients
plt.figure(figsize=(10, 8))
for cluster in data['Cluster'].unique():
    plt.scatter(data[data['Cluster'] == cluster]['Age'], data[data['Cluster'] == cluster]['Health Score'], label=f'Cluster {cluster}')

plt.title('Patient Segmentation')
plt.xlabel('Age')
plt.ylabel('Health Score')
plt.legend()
plt.show()

