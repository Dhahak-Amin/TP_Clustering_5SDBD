import numpy as np
import matplotlib.pyplot as plt
from scipy.io import arff
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.metrics.pairwise import euclidean_distances
import time
import pandas as pd

def evaluate_clustering(X, method, k_values):
    inertias = []
    silhouettes = []
    calinski_scores = []
    davies_scores = []
    best_silhouette = -1
    best_k = None

    for k in k_values:
        if method == 'minibatch':
            model = MiniBatchKMeans(n_clusters=k, init='k-means++', n_init=5, random_state=0)
        else:
            model = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=0)
        labels = model.fit_predict(X)
        inertias.append(model.inertia_)
        sil = silhouette_score(X, labels)
        cal = calinski_harabasz_score(X, labels)
        dav = davies_bouldin_score(X, labels)
        silhouettes.append(sil)
        calinski_scores.append(cal)
        davies_scores.append(dav)
        print(f"k={k:2d} | silhouette={sil:.4f} | calinski={cal:.2f} | davies={dav:.3f} | inertia={model.inertia_:.2f}")
        if sil > best_silhouette:
            best_silhouette = sil
            best_k = k

    # Plot elbow
    plt.figure()
    plt.plot(k_values, inertias, marker='o')
    plt.xlabel('k (number of clusters)')
    plt.ylabel('Inertia')
    plt.title('Elbow Method')
    plt.show()

    # Plot clustering scores
    plt.figure()
    plt.plot(k_values, silhouettes, marker='o', label='Silhouette')
    plt.plot(k_values, calinski_scores, marker='x', label='Calinski-Harabasz')
    plt.plot(k_values, davies_scores, marker='s', label='Davies-Bouldin')
    plt.xlabel('k (number of clusters)')
    plt.title('Clustering Scores')
    plt.legend()
    plt.show()

    print(f"\nBest k according to silhouette: {best_k}")
    return best_k

def plot_best_clustering(X, method, k):
    if method == 'minibatch':
        model = MiniBatchKMeans(n_clusters=k, init='k-means++', n_init=5, random_state=0)
    else:
        model = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=0)
    labels = model.fit_predict(X)
    centroids = model.cluster_centers_
    plt.figure()
    plt.scatter(X[:, 0], X[:, 1], c=labels, s=8, cmap='viridis')
    plt.scatter(centroids[:, 0], centroids[:, 1], marker="x", s=100, linewidths=3, color="red")
    plt.title(f"Clustering {method} (k={k})")
    plt.show()
    print(f"Silhouette: {silhouette_score(X, labels):.4f}")
    print(f"Calinski-Harabasz: {calinski_harabasz_score(X, labels):.2f}")
    print(f"Davies-Bouldin: {davies_bouldin_score(X, labels):.3f}")

if __name__ == "__main__":
    # Load ARFF dataset
    data, meta = arff.loadarff('dataset/artificial/xclara.arff')
    X = np.array([[x[0], x[1]] for x in data])
    name = "xclara.arff"

    # Show initial data
    plt.scatter(X[:, 0], X[:, 1], s=8)
    plt.title(f"Données initiales : {name}")

    plt.show()

    # Find best k
    k_values = range(2, 10)
    method = "kmeans"  # or "minibatch"
    best_k = evaluate_clustering(X, method, k_values)

    # Show best clustering
    plot_best_clustering(X, method, best_k)


