from spine_metrics import SpineMetric, SpineMetricDataset
from typing import List, Iterable, Tuple, Union, Callable, Any, Set
from ipywidgets import widgets
from matplotlib import pyplot as plt
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import euclidean
import numpy as np
from sklearn.decomposition import PCA
from abc import ABC, abstractmethod
import json


class SpineClusterizer(ABC):
    cluster_masks: List[List[bool]]
    num_of_clusters: int
    sample_size: int
    reduced_data: np.ndarray
    pca_dim: int
    _data: np.ndarray
    metric: Union[Callable[[np.ndarray, np.ndarray], float], str]

    def __init__(self, metric: Union[Callable, str] = "euclidean", pca_dim: int = -1):
        self.cluster_masks = []
        self.num_of_clusters = 0
        self.pca_dim = pca_dim
        self.metric = metric

    def get_cluster(self, index: int) -> List[int]:
        return [i for i, x in enumerate(self.cluster_masks[index]) if x]

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
        self.sample_size = spine_metrics.num_of_spines

        self._data = spine_metrics.as_array()
        if self.pca_dim != -1:
            self._data = PCA(self.pca_dim).fit_transform(self._data)

        self.reduced_data = PCA(2).fit_transform(self._data)

        self._fit(self._data)

    def score(self) -> float:
        # TODO: change nan to something sensical
        if self.num_of_clusters < 2 or self.sample_size - 1 < self.num_of_clusters:
            return float("nan")
        labels = self.get_labels()
        # return silhouette_score(self._data, labels, metric=self.metric)
        return silhouette_score(self._data, labels, metric="euclidean")

    @abstractmethod
    def _fit(self, data: np.array) -> object:
        pass

    def show(self, clusters: Set[int] = None) -> widgets.Widget:
        out = widgets.Output()
        with out:
            self._show(clusters)
            plt.show()

        return out

    def _show(self, clusters: Set[int] = None) -> None:
        colors = [plt.cm.Spectral(each) for each in
                  np.linspace(0, 1, len(self.cluster_masks))]
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
                                   distance: Callable = euclidean) -> List[int]:
        # get spines in cluster
        spine_indices = self.get_cluster(cluster_index)
        num_of_samples = min(num_of_samples, len(spine_indices))
        spines = self.reduced_data[spine_indices]
        # calculate center (mean reduced data)
        center = np.mean(spines, 0)
        # calculate distance to center for each spine in cluster
        distances = {}
        for (spine, index) in zip(spines, spine_indices):
            distances[index] = distance(center, spine)
        # sort spines by distance
        spine_indices.sort(key=lambda index: distances[index])
        # return first N spines
        return spine_indices[:num_of_samples]


class ManualSpineClusterizer(SpineClusterizer):
    def __init__(self, cluster_masks: List[List[bool]]):
        super().__init__()
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

    def _sklearn_fit(self, data: np.array) -> object:
        clusterized = DBSCAN(eps=self.eps, min_samples=self.min_samples, metric=self.metric).fit(data)

        # Number of clusters in labels, ignoring noise if present.
        labels = clusterized.labels_
        self.num_of_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        self.num_of_noise = list(labels).count(-1)

        return clusterized

    def _show(self, clusters: Set[int] = None) -> None:
        core_samples_mask = np.zeros_like(self._fit_data.labels_, dtype=bool)
        core_samples_mask[self._fit_data.core_sample_indices_] = True

        # Black removed and is used for noise instead.
        colors = [plt.cm.Spectral(each) for each in
                  np.linspace(0, 1, len(self.cluster_masks))]
        
        labels = self._fit_data.labels_
        for k, col in zip(set(labels), colors):
            if k == -1:
                # Black used for noise.
                col = [0, 0, 0, 1]

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
    def __init__(self, num_of_clusters: int, pca_dim: int = -1):
        super().__init__(pca_dim=pca_dim)
        self.num_of_clusters = num_of_clusters

    def _sklearn_fit(self, data: np.array) -> object:
        return KMeans(n_clusters=self.num_of_clusters, random_state=0).fit(data)
