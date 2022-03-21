from spine_metrics import SpineMetric
from typing import List, Iterable, Tuple, Union, Callable, Any
from ipywidgets import widgets
from matplotlib import pyplot as plt
from sklearn.cluster import DBSCAN, KMeans
import numpy as np
from sklearn.decomposition import PCA
from abc import ABC, abstractmethod


class SpineClusterizer(ABC):
    cluster_masks: List[Iterable[bool]]
    num_of_clusters: int
    _reduced_data: np.ndarray
    _clusterized: object

    def __init__(self):
        self.cluster_masks = []
        self.num_of_clusters = 0

    def fit(self, spine_metrics: Iterable[Iterable[SpineMetric]]) -> None:
        # stack metrics into single vector
        data = []
        for one_spine_metrics in spine_metrics:
            data.append([])
            for spine_metric in one_spine_metrics:
                data[-1] += spine_metric.value_as_list()
        data = np.asarray(data)
        pca = PCA(2)
        self._reduced_data = pca.fit_transform(data)

        self._clusterized = self._fit(data)

        labels = self._clusterized.labels_
        unique_labels = set(labels)
        self.cluster_masks = []
        for k in unique_labels:
            self.cluster_masks.append(labels == k)

    @abstractmethod
    def _fit(self, data: np.array) -> object:
        pass

    def show(self) -> widgets.Widget:
        out = widgets.Output()
        with out:
            self._show()
        return out

    @abstractmethod
    def _show(self) -> None:
        pass


class DBSCANSpineClusterizer(SpineClusterizer):
    eps: float
    min_samples: int
    metric: Union[str, Callable[[np.ndarray, np.ndarray], float]]
    num_of_noise: int

    def __init__(self, eps: float = 0.5, min_samples: int = 2,
                 metric: Union[str, Callable] = "euclidean"):
        super().__init__()
        self.metric = metric
        self.min_samples = min_samples
        self.eps = eps

    def _fit(self, data: np.array) -> object:
        clusterized = DBSCAN(eps=self.eps, min_samples=self.min_samples, metric=self.metric).fit(data)

        # Number of clusters in labels, ignoring noise if present.
        labels = clusterized.labels_
        self.num_of_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        self.num_of_noise = list(labels).count(-1)

        return clusterized

    def _show(self) -> None:
        core_samples_mask = np.zeros_like(self._clusterized.labels_, dtype=bool)
        core_samples_mask[self._clusterized.core_sample_indices_] = True

        # Black removed and is used for noise instead.
        colors = [plt.cm.Spectral(each) for each in
                  np.linspace(0, 1, len(self.cluster_masks))]
        
        labels = self._clusterized.labels_
        for k, col in zip(set(labels), colors):
            if k == -1:
                # Black used for noise.
                col = [0, 0, 0, 1]

            class_member_mask = labels == k

            xy = self._reduced_data[class_member_mask & core_samples_mask]
            if xy.size > 0:
                plt.plot(
                    xy[:, 0],
                    xy[:, 1],
                    "o",
                    markerfacecolor=tuple(col),
                    markeredgecolor="k",
                    markersize=14,
                )

            xy = self._reduced_data[class_member_mask & ~core_samples_mask]
            if xy.size > 0:
                plt.plot(
                    xy[:, 0],
                    xy[:, 1],
                    "o",
                    markerfacecolor=tuple(col),
                    markeredgecolor="k",
                    markersize=6,
                )

        plt.title(f"Estimated number of clusters: {self.num_of_clusters}, eps: {self.eps}")
        plt.show()


class KMeansSpineClusterizer(SpineClusterizer):
    def __init__(self, num_of_clusters: int):
        super().__init__()
        self.num_of_clusters = num_of_clusters

    def _fit(self, data: np.array) -> object:
        return KMeans(n_clusters=self.num_of_clusters).fit(data)

    def _show(self) -> None:
        # Black removed and is used for noise instead.
        colors = [plt.cm.Spectral(each) for each in
                  np.linspace(0, 1, len(self.cluster_masks))]

        labels = self._clusterized.labels_
        for k, col in zip(set(labels), colors):
            if k == -1:
                # Black used for noise.
                col = [0, 0, 0, 1]

            class_member_mask = labels == k

            xy = self._reduced_data[class_member_mask]
            if xy.size > 0:
                plt.plot(
                    xy[:, 0],
                    xy[:, 1],
                    "o",
                    markerfacecolor=tuple(col),
                    markeredgecolor="k",
                    markersize=14,
                )

        plt.title(f"Number of clusters: {self.num_of_clusters}")
        plt.show()
