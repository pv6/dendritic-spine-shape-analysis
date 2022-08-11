from spine_fitter import SpineFitter
from spine_metrics import SpineMetricDataset
from typing import List, Tuple, Union, Callable, Set
from ipywidgets import widgets
from matplotlib import pyplot as plt
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import euclidean
import numpy as np
from sklearn.decomposition import PCA
from abc import ABC, abstractmethod
import json
from scipy.special import kl_div


class SpineClusterizer(SpineFitter, ABC):
    metric: Union[Callable[[np.ndarray, np.ndarray], float], str]
    _data: np.ndarray
    _labels: List[int]

    def __init__(self, metric: Union[Callable, str] = "euclidean", pca_dim: int = -1):
        super().__init__(pca_dim)
        self._labels = []
        self.metric = metric

    @property
    def clusters(self) -> List[Set[str]]:
        return list(self.grouping.groups.values())

    @property
    def outlier_cluster(self) -> Set[str]:
        return self.grouping.outlier_group

    @property
    def num_of_clusters(self) -> int:
        return self.grouping.num_of_groups

    # def score(self) -> float:
    #     # TODO: change nan to something sensical
    #     if self.num_of_clusters < 2 or self.sample_size - 1 < self.num_of_clusters:
    #         return float("nan")
    #     return self._score(self._data, self._labels, metric=self.metric)
    #
    # def _score(self, data, labels, metric):
    #     return silhouette_score(data, labels, metric=metric)

    # def _show(self, groups_to_show: Set[int] = None) -> None:
    #     super()._show(groups_to_show)
    #     # plt.title(f"Number of clusters: {self.num_of_clusters}, score: {self.score():.3f}")
    #     plt.title(f"Number of clusters: {self.num_of_clusters}")


class SKLearnSpineClusterizer(SpineClusterizer, ABC):
    _fit_data: object
    _clusterizer: object

    def _fit(self, data: np.array, names: List[str]) -> None:
        self._fit_data = self._sklearn_fit(data)
        self._labels = self._fit_data.labels_

        for cluster_index in set(self._labels):
            if cluster_index == -1:
                continue
            names_array = np.array(names)
            cluster_names = names_array[self._labels == cluster_index]
            self.grouping.groups[cluster_index + 1] = set(cluster_names)

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

    return output


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

    def __init__(self, eps: float = 0.5, min_samples: int = 2,
                 metric: Union[str, Callable] = "euclidean", pca_dim: int = -1):
        super().__init__(metric=metric, pca_dim=pca_dim)
        self.metric = metric
        self.min_samples = min_samples
        self.eps = eps

    # def score(self) -> float:
    #     # TODO: change nan to something sensical
    #     if self.num_of_outlier / self.sample_size > 0.69 or self.num_of_clusters < 2 or self.sample_size - self.num_of_outlier - 1 < self.num_of_clusters:
    #     # if self.num_of_clusters < 2 or self.sample_size - self.num_of_outlier - 1 < self.num_of_clusters:
    #         return float("nan")
    #     indices_to_delete = np.argwhere(np.asarray(labels) == -1)
    #     filtered_data = np.delete(self._data, indices_to_delete, 0)
    #     filtered_labels = np.delete(self._labels, indices_to_delete, 0)
    #     return self._score(filtered_data, filtered_labels, self.metric)

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
        return clusterized

    # def _show(self, groups_to_show: Set[int] = None) -> None:
    #     core_samples_mask = np.zeros_like(self._fit_data.labels_, dtype=bool)
    #     core_samples_mask[self._fit_data.core_sample_indices_] = True
    #
    #     # Black removed and is used for outlier instead.
    #     colors = self.get_colors()
    #
    #     for k, col in zip(set(self._labels), colors):
    #         class_member_mask = self._labels == k
    #
    #         xy = self.reduced_data[class_member_mask & core_samples_mask]
    #         if xy.size > 0:
    #             plt.plot(
    #                 xy[:, 0],
    #                 xy[:, 1],
    #                 "o",
    #                 markerfacecolor=tuple(col),
    #                 markeredgecolor="k",
    #                 markersize=14,
    #                 label=f"{k}"
    #             )
    #
    #         xy = self.reduced_data[class_member_mask & ~core_samples_mask]
    #         if xy.size > 0:
    #             plt.plot(
    #                 xy[:, 0],
    #                 xy[:, 1],
    #                 "o",
    #                 markerfacecolor=tuple(col),
    #                 markeredgecolor="k",
    #                 markersize=6,
    #                 label=f"{k}"
    #             )
    #     # plt.title(f"Number of clusters: {self.num_of_clusters}, score: {self.score():.3f}")
    #     plt.title(f"Number of clusters: {self.num_of_clusters}")
    #     plt.legend(loc="upper right")


class KMeansSpineClusterizer(SKLearnSpineClusterizer):
    _num_of_clusters: int

    def __init__(self, num_of_clusters: int, pca_dim: int = -1, metric="euclidean"):
        super().__init__(pca_dim=pca_dim, metric=metric)
        self._num_of_clusters = num_of_clusters

    def _sklearn_fit(self, data: np.array) -> object:
        self._clusterizer = KMeans(n_clusters=self._num_of_clusters, random_state=0)
        return self._clusterizer.fit(data)

    # def _score(self, data, labels, metric):
    #     return self._clusterizer.inertia_
