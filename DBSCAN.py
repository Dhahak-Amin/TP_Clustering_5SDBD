import numpy as np
import matplotlib.pyplot as plt
import time

from scipy.io import arff
from sklearn import cluster
from sklearn.neighbors import NearestNeighbors
from sklearn import preprocessing
from sklearn.metrics import silhouette_score

data_path = 'dataset/artificial/'
file_name = "dpc.arff"
raw_data = arff.loadarff(open(data_path + str(file_name), 'r'))
X = np.array([[row[0], row[1]] for row in raw_data[0]])

print("---------------------------------------")
print("Displaying initial data: " + str(file_name))
feature_0 = X[:, 0]
feature_1 = X[:, 1]

# Data standardization
scaler = preprocessing.StandardScaler().fit(X)
X_scaled = scaler.transform(X)
print("Displaying standardized data")
feature_0_scaled = X_scaled[:, 0]
feature_1_scaled = X_scaled[:, 1]


def compute_knn_distances(k):
    neighbors = NearestNeighbors(n_neighbors=k)
    neighbors.fit(X_scaled)
    distances, indices = neighbors.kneighbors(X_scaled)
    avg_distances = np.asarray(
        [np.average(distances[i][1:]) for i in range(distances.shape[0])]
    )
    sorted_distances = np.sort(avg_distances)
    return sorted_distances


def k_nearest_neighbors_plot(k, eps_to_mark=None):
    sorted_distances = compute_knn_distances(k)

    point_start = [0, sorted_distances[0]]
    point_end = [len(sorted_distances) - 1, sorted_distances[-1]]
    distances_to_line = []
    for i in range(len(sorted_distances)):
        x0, y0 = i, sorted_distances[i]
        dist = np.abs(
            (point_end[1] - point_start[1]) * x0
            - (point_end[0] - point_start[0]) * y0
            + point_end[0] * point_start[1]
            - point_end[1] * point_start[0]
        ) / np.sqrt(
            (point_end[1] - point_start[1]) ** 2
            + (point_end[0] - point_start[0]) ** 2
        )
        distances_to_line.append(dist)

    inflection_idx = np.argmax(distances_to_line)
    inflection_x = inflection_idx
    inflection_y = sorted_distances[inflection_idx]

    print('Inflection_x:', inflection_x)
    print('Inflection_y:', inflection_y)

    plt.title(f"{k} Nearest Neighbors - TROUVER LE MEILLEUR EPSILON")
    plt.plot(sorted_distances, label="Sorted distances")
    plt.scatter(inflection_x, inflection_y, color='red', label='Inflection point')
    plt.legend()
    plt.show()

    return inflection_x, inflection_y, sorted_distances


def dbscan_grid_search(sorted_distances, k):
    t_start = time.time()
    best_silhouette = -1
    best_eps = 0.0
    best_min_samples = 0

    eps_min = sorted_distances[0]
    eps_max = sorted_distances[-1]
    eps_values = np.linspace(eps_min, eps_max, 20)

    min_samples_values = range(2, k + 1)

    for eps in eps_values:
        print("eps", round(eps, 3))
        for min_samples in min_samples_values:
            dbscan_model = cluster.DBSCAN(eps=eps, min_samples=min_samples)
            labels = dbscan_model.fit_predict(X_scaled)

            # au moins 2 clusters (hors bruit)
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            if n_clusters < 2:
                continue

            silhouette = silhouette_score(X_scaled, labels)
            if silhouette > best_silhouette:
                best_silhouette = silhouette
                best_eps = eps
                best_min_samples = min_samples

    t_end = time.time()
    print(
        f'Best DBSCAN clustering: eps={best_eps}, '
        f'min_samples={best_min_samples}, silhouette={best_silhouette}'
    )
    return best_eps, best_min_samples, best_silhouette


def plot_dbscan_result(eps, min_samples):
    print("------------------------------------------------------")
    print("Running DBSCAN on standardized data ... ")
    t_start = time.time()
    dbscan_model = cluster.DBSCAN(eps=eps, min_samples=min_samples)
    dbscan_model.fit(X_scaled)
    t_end = time.time()
    labels = dbscan_model.labels_
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)
    print('Number of clusters:', n_clusters)
    print('Number of noise points:', n_noise)
    print(f"DBSCAN execution time: {round((t_end - t_start) * 1000, 2)} ms")
    plt.scatter(feature_0_scaled, feature_1_scaled, c=labels, s=8)
    plt.title(f"DBSCAN clustering - Epsilon={eps} MinPts={min_samples}")
    plt.show()



k = 50

inflection_x, inflection_y, sorted_distances = k_nearest_neighbors_plot(k)

optimal_eps, optimal_min_samples, optimal_silhouette = dbscan_grid_search(
    sorted_distances, k
)
plot_dbscan_result(0.31726745131532974, optimal_min_samples)

k_nearest_neighbors_plot(k, eps_to_mark=optimal_eps)
