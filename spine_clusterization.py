from spine_metrics import SpineMetric, SpineMetricDataset
from typing import List, Iterable, Tuple, Union, Callable, Any, Set
from ipywidgets import widgets
from matplotlib import pyplot as plt
from sklearn.cluster import DBSCAN, KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import euclidean
import numpy as np
from sklearn.decomposition import PCA
from abc import ABC, abstractmethod
import json
from scipy.special import kl_div


class SpineClusterizer(ABC):
    cluster_masks: List[List[bool]]
    num_of_clusters: int
    sample_size: int
    reduced_data: np.ndarray
    pca_dim: int
    _data: np.ndarray
    metric: Union[Callable[[np.ndarray, np.ndarray], float], str]
    dataset: SpineMetricDataset

    def __init__(self, metric: Union[Callable, str] = "euclidean", pca_dim: int = -1):
        self.cluster_masks = []
        self.num_of_clusters = 0
        self.pca_dim = pca_dim
        self.metric = metric
        self.dataset = SpineMetricDataset({})

    def get_cluster(self, index: int) -> Set[str]:
        names = self.dataset.spine_names
        mask = self.cluster_masks[index]
        return {names[j] for j, is_in in enumerate(mask) if is_in}

    def get_clusters(self) -> List[Set[str]]:
        return [self.get_cluster(i) for i in range(self.num_of_clusters)]

    def get_noise_cluster(self) -> Set[str]:
        clusters = self.get_clusters()
        all_names = set(self.dataset.spine_names)
        for cluster in clusters:
            all_names = all_names.difference(cluster)
        return all_names

    def get_labels(self) -> List[int]:
        output = [-1] * self.sample_size
        for i in range(self.sample_size):
            for j in range(self.num_of_clusters):
                if self.cluster_masks[j][i]:
                    output[i] = j
                    break
        return output

    def save_plot(self, filename: str) -> None:
        self._show()
        plt.savefig(filename)
        plt.clf()

    def fit(self, spine_metrics: SpineMetricDataset) -> None:
        self.dataset = spine_metrics

        self.sample_size = spine_metrics.num_of_spines

        self._data = spine_metrics.as_array()
        if self.pca_dim != -1:
            pca = PCA(self.pca_dim)
            self._data = pca.fit_transform(self._data)

        self.reduced_data = PCA(2).fit_transform(self._data)

        self._fit(self._data)

    def score(self) -> float:
        # TODO: change nan to something sensical
        labels = self.get_labels()
        if self.num_of_clusters < 2 or self.sample_size - 1 < self.num_of_clusters:
            return float("nan")
        labels = self.get_labels()
        return self._score(self._data, labels, metric=self.metric)

    def _score(self, data, labels, metric):
        return silhouette_score(data, labels, metric=metric)

    @abstractmethod
    def _fit(self, data: np.array) -> object:
        pass

    def show(self, clusters: Set[int] = None) -> widgets.Widget:
        out = widgets.Output()
        with out:
            self._show(clusters)
            plt.show()

        return out

    def get_colors(self) -> List[Tuple[float, float, float, float]]:
        return [plt.cm.Spectral(each) for each in
                np.linspace(0, 1, self.num_of_clusters)]

    def _show(self, clusters: Set[int] = None) -> None:
        colors = self.get_colors()
        for i, (mask, color) in enumerate(zip(self.cluster_masks, colors)):
            if clusters is not None and i not in clusters:
                color = [0.69, 0.69, 0.69, 1]
            xy = self.reduced_data[mask]
            if xy.size > 0:
                plt.plot(
                    xy[:, 0],
                    xy[:, 1],
                    "o",
                    markerfacecolor=tuple(color),
                    markeredgecolor="k",
                    markersize=14,
                    label=f"{i}"
                )
        plt.title(f"Number of clusters: {self.num_of_clusters}, score: {self.score():.3f}")
        plt.legend(loc="upper right")

    def save(self, filename: str) -> None:
        with open(filename, "w") as file:
            json.dump({"cluster_masks": self.cluster_masks}, file)

    def get_representative_samples(self, cluster_index: int,
                                   num_of_samples: int = 4,
                                   distance: Callable = euclidean) -> List[str]:
        if distance is None:
            distance = euclidean
        
        # get spines in cluster
        spine_names = self.get_cluster(cluster_index)
        spine_indices = [self.dataset.get_row_index(name) for name in spine_names]
        num_of_samples = min(num_of_samples, len(spine_indices))
        spines = self.reduced_data[spine_indices]
        # calculate center (mean reduced data)
        center = np.mean(spines, 0)
        # calculate distance to center for each spine in cluster
        distances = {}
        for (spine, index) in zip(spines, spine_indices):
            distances[index] = distance(center, spine)
        # sort spines by distance
        output = list(zip(spine_names, spine_indices))
        output.sort(key=lambda name_index: distances[name_index[1]])
        # return first N spine names
        return [name_index[0] for name_index in output[:num_of_samples]]


class ManualSpineClusterizer(SpineClusterizer):
    def __init__(self, cluster_masks: List[List[bool]],
                 metric: Union[str, Callable] = "euclidean"):
        super().__init__(metric=metric)
        self.cluster_masks = cluster_masks
        self.num_of_clusters = len(cluster_masks)
        if self.num_of_clusters > 0:
            self.num_of_samples = len(cluster_masks[0])
        else:
            self.num_of_samples = 0

    def _fit(self, data: np.array) -> object:
        pass


def load_clusterization(filename: str) -> SpineClusterizer:
    masks = []
    with open(filename) as file:
        masks = json.load(file)["cluster_masks"]
    return ManualSpineClusterizer(masks)


class SKLearnSpineClusterizer(SpineClusterizer, ABC):
    _fit_data: object
    _clusterizer: object

    def _fit(self, data: np.array) -> None:
        self._fit_data = self._sklearn_fit(data)
        labels = self._fit_data.labels_
        unique_labels = set(labels)
        self.cluster_masks = []
        for k in unique_labels:
            self.cluster_masks.append([bool(label == k) for label in labels])

    @abstractmethod
    def _sklearn_fit(self, data: np.ndarray) -> object:
        pass


def ks_test(x: np.ndarray, y: np.ndarray) -> float:
    output = 0
    sum_x = 0
    sum_y = 0
    for i in range(x.size):
        sum_x += x[i]
        sum_y += y[i]
        output = max(output, abs(sum_x - sum_y))

    return output / x.size / 100


def chi_square_distance(x: np.ndarray, y: np.ndarray) -> float:
    return 0.5 * np.sum(((x - y) ** 2) / (x + y))


def symmetric_kl_div(x: np.ndarray, y: np.ndarray) -> float:
    x += np.ones_like(x) * 0.001
    x /= np.sum(x)
    y += np.ones_like(x) * 0.001
    y /= np.sum(x)

    a = kl_div(x, y)
    b = kl_div(y, x)

    s = np.sum((a + b) / 2)
    return float(s) / x.size


class DBSCANSpineClusterizer(SKLearnSpineClusterizer):
    eps: float
    min_samples: int
    num_of_noise: int

    def __init__(self, eps: float = 0.5, min_samples: int = 2,
                 metric: Union[str, Callable] = "euclidean", pca_dim: int = -1):
        super().__init__(metric=metric, pca_dim=pca_dim)
        self.metric = metric
        self.min_samples = min_samples
        self.eps = eps

    def score(self) -> float:
        # TODO: change nan to something sensical
        if self.num_of_noise / self.sample_size > 0.69 or self.num_of_clusters < 2 or self.sample_size - self.num_of_noise - 1 < self.num_of_clusters:
        # if self.num_of_clusters < 2 or self.sample_size - self.num_of_noise - 1 < self.num_of_clusters:
            return float("nan")
        labels = self.get_labels()
        indices_to_delete = np.argwhere(np.asarray(labels) == -1)
        filtered_data = np.delete(self._data, indices_to_delete, 0)
        filtered_labels = np.delete(labels, indices_to_delete, 0)
        return self._score(filtered_data, filtered_labels, self.metric)

    # def _score(self, data, labels, metric):
    #     neigh = NearestNeighbors(n_neighbors=2, metric=metric)
    #     nbrs = neigh.fit(self._data)
    #     distances, indices = nbrs.kneighbors(self._data)
    #     distances = -np.sort(-distances, axis=0)
    #     distances = distances[:, 1]
    #     for i in range(len(distances)):
    #         if self.eps > distances[i]:
    #             return i
    #     return len(distances)

    def _sklearn_fit(self, data: np.array) -> object:
        self._clusterizer = DBSCAN(eps=self.eps, min_samples=self.min_samples, metric=self.metric)
        clusterized = self._clusterizer.fit(data)

        # Number of clusters in labels, ignoring noise if present.
        labels = clusterized.labels_
        self.num_of_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        self.num_of_noise = list(labels).count(-1)

        return clusterized

    def get_colors(self) -> List[Tuple[float, float, float, float]]:
        colors = [plt.cm.Spectral(each) for each in
                  np.linspace(0, 1, len(self.cluster_masks))]
        labels = self._fit_data.labels_
        for i, k in enumerate(set(labels)):
            if k == -1:
                # Black used for noise.
                colors[i] = [0, 0, 0, 1]
                break
        return colors

    def _show(self, clusters: Set[int] = None) -> None:
        core_samples_mask = np.zeros_like(self._fit_data.labels_, dtype=bool)
        core_samples_mask[self._fit_data.core_sample_indices_] = True

        # Black removed and is used for noise instead.
        colors = self.get_colors()
        
        labels = self._fit_data.labels_
        for k, col in zip(set(labels), colors):
            class_member_mask = labels == k

            xy = self.reduced_data[class_member_mask & core_samples_mask]
            if xy.size > 0:
                plt.plot(
                    xy[:, 0],
                    xy[:, 1],
                    "o",
                    markerfacecolor=tuple(col),
                    markeredgecolor="k",
                    markersize=14,
                    label=f"{k}"
                )

            xy = self.reduced_data[class_member_mask & ~core_samples_mask]
            if xy.size > 0:
                plt.plot(
                    xy[:, 0],
                    xy[:, 1],
                    "o",
                    markerfacecolor=tuple(col),
                    markeredgecolor="k",
                    markersize=6,
                    label=f"{k}"
                )
        plt.title(f"Number of clusters: {self.num_of_clusters}, score: {self.score():.3f}")
        plt.legend(loc="upper right")


class KMeansSpineClusterizer(SKLearnSpineClusterizer):
    def __init__(self, num_of_clusters: int, pca_dim: int = -1, metric="euclidean"):
        super().__init__(pca_dim=pca_dim, metric=metric)
        self.num_of_clusters = num_of_clusters

    def _sklearn_fit(self, data: np.array) -> object:
        self._clusterizer = KMeans(n_clusters=self.num_of_clusters, random_state=0)
        return self._clusterizer.fit(data)

    # def _score(self, data, labels, metric):
    #     return self._clusterizer.inertia_
