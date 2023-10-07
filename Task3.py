"""#Task3"""

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
import umap
import matplotlib.pyplot as plt

#before runing this step mount the drive if you are running through Colab

data_movies = "/content/drive/MyDrive/Assianments/Triveni_CA2/IMDB Dataset.csv"

df_movies = pd.read_csv(data_movies)

df_movies.head()

df_movies.shape

df_movies.isnull().sum()

# Create a TF-IDF matrix
tfidf_vectorizer = TfidfVectorizer(max_features=5000)
tfidf_matrix = tfidf_vectorizer.fit_transform(df_movies['review'])

# Perform PCA
pca = PCA(n_components=2)
pca_result = pca.fit_transform(tfidf_matrix.toarray())

# Perform UMAP
umap_model = umap.UMAP(n_components=2)
umap_result = umap_model.fit_transform(tfidf_matrix)

# Adding sentiment labels to the results
pca_result = pd.DataFrame(pca_result, columns=['PC1', 'PC2'])
umap_result = pd.DataFrame(umap_result, columns=['UMAP1', 'UMAP2'])
results = pd.concat([pca_result, umap_result, df_movies['sentiment']], axis=1)

# Plotting the results
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(pca_result['PC1'], pca_result['PC2'], c=df_movies['sentiment'].map({'positive': 'blue', 'negative': 'red'}), alpha=0.5)
plt.title('PCA Visualization')

plt.subplot(1, 2, 2)
plt.scatter(umap_result['UMAP1'], umap_result['UMAP2'], c=df_movies['sentiment'].map({'positive': 'blue', 'negative': 'red'}), alpha=0.5)
plt.title('UMAP Visualization')

plt.tight_layout()
plt.show()

# Perform UMAP for 3D visualization
umap_model_3d = umap.UMAP(n_components=3)
umap_result_3d = umap_model_3d.fit_transform(tfidf_matrix.toarray())

# 3D UMAP Visualization
ax = plt.subplot(1, 3, 3, projection='3d')
ax.scatter(umap_result_3d[:, 0], umap_result_3d[:, 1], umap_result_3d[:, 2], c=df_movies['sentiment'].map({'positive': 'blue', 'negative': 'red'}), alpha=0.5)
ax.set_title('UMAP 3D Visualization')

plt.tight_layout()
plt.show()

# Create interactive 3D UMAP plot using Plotly
fig = px.scatter_3d(df_movies, x=umap_result_3d[:, 0], y=umap_result_3d[:, 1], z=umap_result_3d[:, 2], color='sentiment', color_discrete_map={'positive': 'blue', 'negative': 'red'}, opacity=0.7)
fig.update_layout(title='Interactive 3D UMAP Visualization for Movie Reviews Sentiment', legend_title='Sentiment')

# Display the plot
fig.show()