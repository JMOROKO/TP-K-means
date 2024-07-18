import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

data = np.random.rand(100, 5)

print("Taille des données : ", data.shape)

random_kmeans = KMeans(n_clusters=3, init='random', n_init=10)
random_kmeans.fit(data)

kmeans_plus = KMeans(n_clusters=3, init='k-means++', n_init=10)
kmeans_plus.fit(data)

avg = silhouette_score(data, kmeans_plus.labels_)
index = calinski_harabasz_score(data, kmeans_plus.labels_)
print("Silhouette Score : ", avg)
print("Calinski-Harabasz Index : ", index)

if silhouette_score(data, random_kmeans.labels_) > avg:
    best_model = random_kmeans
else:
    best_model = kmeans_plus

centers = best_model.cluster_centers_

pca = PCA(n_components=2)
data_pca = pca.fit_transform(data)

print("Valeurs propres : ", pca.explained_variance_)
print("Vecteurs propres : ", pca.components_)

print("Inertie de chaque axe : ", pca.explained_variance_ratio_)

print("Somme des inerties : ", np.sum(pca.explained_variance_ratio_))

plt.scatter(data_pca[:, 0], data_pca[:, 1], c=best_model.labels_)
plt.scatter(pca.transform(centers)[:, 0], pca.transform(centers)[:, 1], marker='x', c='red')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('Clustering avec PCA')
plt.show()

