# -*- coding: utf-8 -*-
"""Task2
Original file is located at
    https://colab.research.google.com/drive/16Rx7H4-RY6XVd-YGfgH-k9Zk34F2pJZR

#Task2
"""

#first install the umap than the package will install desired libaries
#!pip install umap-learn

import pandas as pd
import numpy as np
import umap
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Loading the dataset from the URL
data_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00468/online_shoppers_intention.csv"
df = pd.read_csv(data_url)

df.head()

df.info()

df.shape

df.describe()

df.isnull().sum()

df['Revenue']

# Select the relevant features
features = df[['Administrative', 'Informational', 'ProductRelated', 'Administrative_Duration', 'Informational_Duration', 'ProductRelated_Duration']]

# Standardize the features
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# Perform PCA
pca = PCA(n_components=3)
pca_result = pca.fit_transform(scaled_features)

# Perform UMAP
umap_model = umap.UMAP(n_components=3)
umap_result = umap_model.fit_transform(scaled_features)

# Create plots for PCA and UMAP
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(pca_result[:, 0], pca_result[:, 1], c=df['Revenue'], cmap='coolwarm', alpha=0.5)
plt.title('PCA Visualization')

plt.subplot(1, 2, 2)
plt.scatter(umap_result[:, 0], umap_result[:, 1], c=df['Revenue'], cmap='coolwarm', alpha=0.5)
plt.title('UMAP Visualization')

plt.tight_layout()
plt.show()



# For PCA
best_pca_result = None
best_pca_variance = 0
for n_components in range(2, 7):
    pca = PCA(n_components=n_components)
    pca_result = pca.fit_transform(scaled_features)
    if pca.explained_variance_ratio_.sum() > best_pca_variance:
        best_pca_result = pca_result
        best_pca_variance = pca.explained_variance_ratio_.sum()

# Create a DataFrame for the interactive plot
df_pca = pd.DataFrame(data=best_pca_result, columns=[f'PCA Dimension {i+1}' for i in range(best_pca_result.shape[1])])

# Add a column for coloring the points by target
df_pca['Target'] = df['Revenue']

# Create an interactive scatter plot using Plotly
fig_pca = px.scatter(df_pca, x='PCA Dimension 1', y='PCA Dimension 2', color='Target', color_continuous_scale='coolwarm', opacity=0.5)
fig_pca.update_layout(title='Interactive PCA Visualization for Online Shoppers Dataset', xaxis_title='PCA Dimension 1', yaxis_title='PCA Dimension 2')

# Display the interactive plot
fig_pca.show()

# Create a DataFrame for the interactive 3D plot
df_pca_3d = pd.DataFrame(data=best_pca_result, columns=[f'PCA Dimension {i+1}' for i in range(best_pca_result.shape[1])])

# Add a column for coloring the points by target
df_pca_3d['Target'] = df['Revenue']

# Create an interactive 3D scatter plot using Plotly
fig_pca_3d = px.scatter_3d(df_pca_3d, x='PCA Dimension 1', y='PCA Dimension 2', z='PCA Dimension 3', color='Target', color_continuous_scale='coolwarm', opacity=0.5)
fig_pca_3d.update_layout(title='Interactive 3D PCA Visualization for Online Shoppers Dataset')

# Display the interactive 3D plot
fig_pca_3d.show()

# Perform UMAP with hyperparameter tuning
best_umap_result = None
best_umap_score = float('inf')
for n_neighbors in [5, 10, 15]:
    for min_dist in [0.1, 0.5, 0.9]:
        umap_model = umap.UMAP(n_components=2, n_neighbors=n_neighbors, min_dist=min_dist)
        umap_result = umap_model.fit_transform(scaled_features)
        # No reconstruction_error() for UMAP
        # Use UMAP score (negative log-likelihood) for comparison
        umap_score = umap_model._a
        if umap_score < best_umap_score:
            best_umap_result = umap_result
            best_umap_score = umap_score

# Visualize the UMAP results
plt.scatter(best_umap_result[:, 0], best_umap_result[:, 1], c=df['Revenue'], cmap='coolwarm', alpha=0.5)
plt.title('UMAP Visualization for Online Shoppers Dataset')
plt.xlabel('UMAP Dimension 1')
plt.ylabel('UMAP Dimension 2')
plt.show()

#for interactive plots
import plotly.express as px

# Create an interactive Plotly scatter plot
data = pd.DataFrame(data=best_umap_result, columns=['UMAP Dimension 1', 'UMAP Dimension 2'])
data['Revenue'] = df['Revenue']

fig = px.scatter(data, x='UMAP Dimension 1', y='UMAP Dimension 2', color='Revenue', color_continuous_scale='coolwarm', opacity=0.5)
fig.update_layout(title='Interactive UMAP Visualization for Online Shoppers Dataset', xaxis_title='UMAP Dimension 1', yaxis_title='UMAP Dimension 2')

# Display the interactive plot
fig.show()

# Perform UMAP with hyperparameter tuning
best_umap_result = None
best_umap_score = float('inf')
for n_neighbors in [5, 10, 15]:
    for min_dist in [0.1, 0.5, 0.9]:
        umap_model = umap.UMAP(n_components=3, n_neighbors=n_neighbors, min_dist=min_dist)
        umap_result = umap_model.fit_transform(scaled_features)
        # No reconstruction_error() for UMAP
        # Use UMAP score (negative log-likelihood) for comparison
        umap_score = umap_model._a
        if umap_score < best_umap_score:
            best_umap_result = umap_result
            best_umap_score = umap_score

# Create a DataFrame for the interactive 3D plot
df_umap_3d = pd.DataFrame(data=umap_result, columns=['UMAP Dimension 1', 'UMAP Dimension 2', 'UMAP Dimension 3'])
df_umap_3d['Revenue'] = df['Revenue']

# Create an interactive 3D scatter plot using Plotly
fig_umap_3d = px.scatter_3d(df_umap_3d, x='UMAP Dimension 1', y='UMAP Dimension 2', z='UMAP Dimension 3', color='Revenue', color_continuous_scale='coolwarm', opacity=0.5)
fig_umap_3d.update_layout(title='Interactive 3D UMAP Visualization for Online Shoppers Dataset')

# Display the interactive 3D plot
fig_umap_3d.show()
