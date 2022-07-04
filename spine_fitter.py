from spine_metrics import SpineMetricDataset
from typing import List, Tuple, Callable, Set, Dict, Any
from ipywidgets import widgets
from matplotlib import pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from abc import ABC, abstractmethod
import json


class SpineGrouping:
    groups: Dict[Any, Set[str]]
    samples: Set[str]

    def __init__(self, samples: Set[str] = None, groups: Dict[Any, Set[str]] = None):
        if samples is None:
            samples = set()
        if groups is None:
            groups = {}
        self.samples = samples
        self.groups = groups

    @property
    def num_of_groups(self) -> int:
        return len(self.groups)

    @property
    def group_labels(self) -> Set[Any]:
        return set(self.groups.keys())

    @property
    def outlier_group(self) -> Set[str]:
        ng = self.samples
        for group in self.groups.values():
            ng = ng.difference(group)
        return ng

    @property
    def num_of_outlier(self) -> int:
        return len(self.samples) - len(self.outlier_group)

    @property
    def sample_size(self) -> int:
        return len(self.samples)

    @property
    def colors(self) -> Dict[Any, Tuple[float, float, float, float]]:
        label_each = zip(self.groups.keys(),
                         np.linspace(0, 1, self.num_of_groups))
        return {group_label: plt.cm.Spectral(each)
                for (group_label, each) in label_each}

    def save(self, filename: str) -> None:
        with open(filename, "w") as file:
            json.dump({"groups": {label: list(group) for (label, group) in self.groups.items()},
                       "samples": list(self.samples)}, file)

    def load(self, filename: str) -> None:
        with open(filename) as file:
            loaded = json.load(file)
            self.samples = set(loaded["samples"])
            self.groups = loaded["groups"]
            for (key, group) in self.groups.items():
                self.groups[key] = set(group)

    def get_group(self, spine_name: str) -> Any:
        for group_label, group in self.groups.items():
            if spine_name in group:
                return group_label
        return None

    def get_color(self, spine_name: str) -> Tuple[float, float, float, float]:
        group_label = self.get_group(spine_name)
        if group_label is not None:
            return self.colors[group_label]
        return 0.30, 0.30, 0.30, 1

    def show(self, metrics: SpineMetricDataset,
             groups_to_show: Set[int] = None) -> widgets.Widget:
        out = widgets.Output()
        with out:
            self._show(metrics, groups_to_show)
            plt.show()

        return out

    def save_plot(self, metrics: SpineMetricDataset, filename: str) -> None:
        self._show(metrics)
        plt.savefig(filename)
        plt.clf()

    def _show(self, metrics: SpineMetricDataset,
              groups_to_show: Set[Any] = None) -> None:
        def show_group(group_label: Any, group: Set[str],
                       color: Tuple[float, float, float, float]) -> None:
            xy = reduced_data[[metrics.get_row_index(name) for name in group]]
            if xy.size > 0:
                plt.plot(
                    xy[:, 0],
                    xy[:, 1],
                    "o",
                    markerfacecolor=tuple(color),
                    markeredgecolor="k",
                    markersize=14,
                    label=f"{group_label}"
                )

        if groups_to_show is None:
            groups_to_show = set(self.groups.keys())

        colors = self.colors
        reduced_data = PCA(2).fit_transform(metrics.as_array())

        for (group_label, group) in self.groups.items():
            color = colors[group_label] if group_label in groups_to_show else [
                0.69, 0.69, 0.69, 1]
            show_group(group_label, group, color)
        show_group("Outliers", self.outlier_group, (0, 0, 0, 1))

        plt.title(f"Number of groups: {self.num_of_groups}")
        plt.legend(loc="upper right")


class SpineFitter(ABC):
    grouping: SpineGrouping
    pca_dim: int

    def __init__(self, pca_dim: int = -1):
        self.pca_dim = pca_dim
        self.grouping = SpineGrouping()

    # def get_representative_samples(self, group_index: int,
    #                                num_of_samples: int = 4,
    #                                distance: Callable = euclidean) -> List[str]:
    #     if distance is None:
    #         distance = euclidean
    #
    #     # get spines in cluster
    #     spine_names = self.groups[group_index]
    #     num_of_samples = min(num_of_samples, len(spine_names))
    #     spines = [self.get_spine_reduced_coord(name) for name in spine_names]
    #     # calculate center (mean reduced data)
    #     center = np.mean(spines, 0)
    #     # calculate distance to center for each spine in cluster
    #     distances = {}
    #     for (spine, name) in zip(spines, spine_names):
    #         distances[name] = distance(center, spine)
    #     # sort spines by distance
    #     output = list(spine_names)
    #     output.sort(key=lambda name: distances[name])
    #     # return first N spine names
    #     return output[:num_of_samples]

    def fit(self, spine_metrics: SpineMetricDataset) -> None:
        data = spine_metrics.as_array()
        if self.pca_dim != -1:
            pca = PCA(self.pca_dim)
            data = pca.fit_transform(data)

        self.grouping.samples = spine_metrics.spine_names

        self._fit(data, list(spine_metrics.spine_names))

    @abstractmethod
    def _fit(self, data: np.array, names: List[str]) -> object:
        pass
