import numpy as np
import matplotlib.pyplot as plt
import time

from scipy.io import arff
from sklearn import preprocessing
from sklearn.metrics import silhouette_score
import hdbscan


data_path = 'dataset/artificial/'
file_name = "insa.arff"
raw_data = arff.loadarff(open(data_path + str(file_name), 'r'))
X = np.array([[row[0], row[1]] for row in raw_data[0]])

print("---------------------------------------")
print("Displaying initial data: " + str(file_name))
feature_0 = X[:, 0]
feature_1 = X[:, 1]
plt.scatter(feature_0, feature_1, s=8)
plt.title("Initial data: " + str(file_name))
plt.show()

scaler = preprocessing.StandardScaler().fit(X)
X_scaled = scaler.transform(X)
print("Displaying standardized data")
feature_0_scaled = X_scaled[:, 0]
feature_1_scaled = X_scaled[:, 1]
plt.scatter(feature_0_scaled, feature_1_scaled, s=8)
plt.title("Standardized data")
plt.show()


def plot_hdbscan_result(min_cluster_size):
    print("------------------------------------------------------")
    print(
        f"Running HDBSCAN with "
        f"min_cluster_size={min_cluster_size}"
    )
    t_start = time.time()
    hdb = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size
    )
    hdb.fit(X_scaled)
    t_end = time.time()

    labels = hdb.labels_
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)

    print('Number of clusters:', n_clusters)
    print('Number of noise points:', n_noise)
    print(f"HDBSCAN execution time: {round((t_end - t_start)*1000, 2)} ms")

    if n_clusters > 1:
        silhouette = silhouette_score(X_scaled, labels)
        print(f"Silhouette score: {silhouette:.4f}")
    else:
        silhouette = -1
        print("Silhouette score: Not defined (only one cluster)")

    plt.scatter(feature_0_scaled, feature_1_scaled, c=labels, s=8)
    plt.title(
        f"HDBSCAN clustering - "
        f"min_cluster_size={min_cluster_size}")
    plt.show()

    return silhouette


def hdbscan_grid_search():
    t_start = time.time()
    best_silhouette = -1
    best_min_cluster_size = None

    # on teste différentes tailles de clusters
    for min_cluster_size in range(2, 30):
        print("HDBSCAN - testing min_cluster_size =", min_cluster_size)
        hdb = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
          
        )
        labels = hdb.fit_predict(X_scaled)

        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        if n_clusters <= 1:
            continue

        silhouette = silhouette_score(X_scaled, labels)
        if silhouette > best_silhouette:
            best_silhouette = silhouette
            best_min_cluster_size = min_cluster_size

    t_end = time.time()
    print(
        f'Best HDBSCAN clustering: '
        f'min_cluster_size={best_min_cluster_size}, '
        f'silhouette={best_silhouette:.4f}'
    )
    print(
        f"HDBSCAN grid search execution time: "
        f"{round((t_end - t_start)*1000, 2)} ms"
    )
    return best_min_cluster_size, best_silhouette

optimal_min_cluster_size, optimal_silhouette = hdbscan_grid_search()
plot_hdbscan_result(optimal_min_cluster_size)
