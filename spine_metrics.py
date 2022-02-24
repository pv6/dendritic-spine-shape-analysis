from abc import ABC, abstractmethod
from random import Random
import numpy as np
from typing import Any, List, Tuple, Dict
import ipywidgets as widgets
from CGAL.CGAL_Polyhedron_3 import Polyhedron_3, Polyhedron_3_Facet_handle
from CGAL.CGAL_Polygon_mesh_processing import area, volume
from CGAL.CGAL_Kernel import Ray_3, Point_3, Vector_3
from CGAL.CGAL_AABB_tree import AABB_tree_Polyhedron_3_Facet_handle
from spine_segmentation import point_2_list
from matplotlib import pyplot as plt
import csv


class SpineMetric(ABC):
    name: str
    value: Any

    def __init__(self, spine_mesh: Polyhedron_3) -> None:
        self.value = self._calculate(spine_mesh)

    @abstractmethod
    def _calculate(self, spine_mesh: Polyhedron_3) -> Any:
        pass

    def show(self) -> None:
        pass

    def value_as_list(self) -> List[Any]:
        try:
            return [*self.value]
        except TypeError:
            return [self.value]


class CustomSpineMetric(SpineMetric):
    def __init__(self, name: str, value: Any) -> None:
        super().__init__(Polyhedron_3())
        self.value = value
        self.name = name

    def _calculate(self, spine_mesh: Polyhedron_3) -> Any:
        pass


def calculate_metrics(spine_mesh: Polyhedron_3,
                      metric_names: List[str]) -> List[SpineMetric]:
    out = []
    for name in metric_names:
        klass = globals()[name + "SpineMetric"]
        out.append(klass(spine_mesh))
    return out


SPINE_FILE_FIELD = "Spine File"


def save_metrics(metrics: Dict[str, List[SpineMetric]], filename: str) -> None:
    with open(filename, mode="w") as file:
        if len(metrics) == 0:
            return
        # extract from metric names from first spine
        metric_names = [metric.name for metric in list(metrics.values())[0]]

        # save metrics for each spine
        writer = csv.DictWriter(file, fieldnames=[SPINE_FILE_FIELD] + metric_names)
        writer.writeheader()
        for spine_name in metrics.keys():
            writer.writerow({SPINE_FILE_FIELD: spine_name,
                             **{metric.name: metric.value
                                for metric in metrics[spine_name]}})


def load_metrics(filename: str) -> Dict[str, List[SpineMetric]]:
    output = {}
    with open(filename, mode="r") as file:
        reader = csv.DictReader(file)
        for row in reader:
            # extract spine file name
            spine_name = row.pop(SPINE_FILE_FIELD)
            # extract each metric
            metrics = []
            for metric_name in row.keys():
                value_str = row[metric_name]
                value = None
                if value_str[0] == "[":
                    value = np.fromstring(value_str[1:-1], dtype="float", sep=" ")
                else:
                    value = float(value_str)
                metrics.append(CustomSpineMetric(metric_name, value))
            output[spine_name] = metrics
    return output


class FloatSpineMetric(SpineMetric, ABC):
    def show(self):
        return widgets.Label(f"{self.value:.2f}")


class AreaSpineMetric(FloatSpineMetric):
    def __init__(self, spine_mesh: Polyhedron_3) -> None:
        self.name = "Area"
        super().__init__(spine_mesh)

    def _calculate(self, spine_mesh: Polyhedron_3) -> Any:
        return area(spine_mesh)


class VolumeSpineMetric(FloatSpineMetric):
    def __init__(self, spine_mesh: Polyhedron_3) -> None:
        self.name = "Volume"
        super().__init__(spine_mesh)

    def _calculate(self, spine_mesh: Polyhedron_3) -> Any:
        return volume(spine_mesh)


class HistogramSpineMetric(SpineMetric):
    num_of_bins: int
    distribution: np.array

    def __init__(self, spine_mesh: Polyhedron_3, num_of_bins: int = 10) -> None:
        self.num_of_bins = num_of_bins
        super().__init__(spine_mesh)

    def show(self):
        out = widgets.Output()

        #get_ipython().magic("matplotlib inline")

        with out:
            plt.hist(self.distribution, self.num_of_bins, density=True)
            plt.show()

        return out

    @abstractmethod
    def _calculate_distribution(self, spine_mesh: Polyhedron_3) -> np.array:
        pass

    def _calculate(self, spine_mesh: Polyhedron_3) -> np.array:
        self.distribution = self._calculate_distribution(spine_mesh)
        return np.histogram(self.distribution, self.num_of_bins, density=True)[0]


class ChordDistributionSpineMetric(HistogramSpineMetric):
    num_of_chords: int
    chords: List[Tuple[Point_3, Point_3]]
    chord_lengths: List[float]

    def __init__(self, spine_mesh: Polyhedron_3, num_of_chords: int = 1000, num_of_bins: int = 20) -> None:
        self.name = "Chord Distribution"
        self.num_of_chords = num_of_chords
        super().__init__(spine_mesh, num_of_bins)

    @staticmethod
    def _calculate_facet_center(facet: Polyhedron_3_Facet_handle) -> Vector_3:
        circulator = facet.facet_begin()
        begin = facet.facet_begin()
        center = Vector_3(0, 0, 0)
        while circulator.hasNext():
            halfedge = circulator.next()
            pnt = halfedge.vertex().point()
            center += Vector_3(pnt.x(), pnt.y(), pnt.z())
            # check for end of loop
            if circulator == begin:
                break
        center /= 3
        return center

    @staticmethod
    def _vec_2_point(vector: Vector_3) -> Point_3:
        return Point_3(vector.x(), vector.y(), vector.z())

    @staticmethod
    def _point_2_vec(point: Point_3) -> Vector_3:
        return Vector_3(point.x(), point.y(), point.z())

    def _calculate_raycast(self, ray_query: Ray_3,
                           tree: AABB_tree_Polyhedron_3_Facet_handle) -> None:
        intersections = []
        tree.all_intersections(ray_query, intersections)

        origin = self._point_2_vec(ray_query.source())
        prev_dist = 0

        # sort intersections along the ray
        intersection_points = [self._calculate_facet_center(intersection.second)
                               for intersection in intersections]
        intersection_points.sort(key=lambda point: (point - origin).squared_length())

        for j in range(1, len(intersection_points), 2):
            center_1 = intersection_points[j - 1]
            center_2 = intersection_points[j]

            # # check intersections are in correct order along the ray
            # dist = np.sqrt((center_1 - origin).squared_length())
            # if dist > prev_dist:
            #     prev_dist = dist
            #     continue
            # prev_dist = dist

            self.chord_lengths.append(
                np.sqrt((center_2 - center_1).squared_length()))

            self.chords.append(
                (self._vec_2_point(center_1), self._vec_2_point(center_2)))

    def _calculate_distribution(self, spine_mesh: Polyhedron_3) -> np.array:
        self.chord_lengths = []
        self.chords = []

        tree = AABB_tree_Polyhedron_3_Facet_handle(spine_mesh.facets())

        surface_points = [self._calculate_facet_center(facet) for facet in spine_mesh.facets()]

        rand = Random()
        for i in range(self.num_of_chords):
            ind1 = rand.randrange(0, len(surface_points))
            ind2 = rand.randrange(0, len(surface_points))
            while ind1 == ind2:
                ind2 = rand.randrange(0, len(surface_points))
            p1 = surface_points[ind1]
            p2 = surface_points[ind2]
            direction = p2 - p1
            direction.normalize()

            ray_query = Ray_3(self._vec_2_point(p1 - direction * 500),
                              self._vec_2_point(p2 + direction * 500))
            self._calculate_raycast(ray_query, tree)

        # # find bounding box
        # points = [point_2_list(point) for point in spine_mesh.points()]
        # min_coord = np.min(points, axis=0)
        # max_coord = np.max(points, axis=0)
        #
        # min_coord -= [0.1, 0.1, 0.1]
        # max_coord += [0.1, 0.1, 0.1]
        #
        # step = (max_coord - min_coord) / (self.num_of_chords + 1)
        #
        # for x in range(1, self.num_of_chords + 1):
        #     for y in range(1, self.num_of_chords + 1):
        #         # ray generation
        #         # left to right
        #         ray_query = Ray_3(Point_3(min_coord[0], min_coord[1] + x * step[1], min_coord[2] + y * step[2]),
        #                           Point_3(max_coord[0], min_coord[1] + x * step[1], min_coord[2] + y * step[2]))
        #         self._calculate_raycast(ray_query, tree)
        #         # down to up
        #         ray_query = Ray_3(Point_3(min_coord[0] + x * step[0], min_coord[1], min_coord[2] + y * step[2]),
        #                           Point_3(min_coord[0] + x * step[0], max_coord[1], min_coord[2] + y * step[2]))
        #         self._calculate_raycast(ray_query, tree)
        #         # backward to forward
        #         ray_query = Ray_3(Point_3(min_coord[0] + x * step[0], min_coord[1] + y * step[1], min_coord[2]),
        #                           Point_3(min_coord[0] + x * step[0], min_coord[1] + y * step[1], max_coord[2]))
        #         self._calculate_raycast(ray_query, tree)

        max_len = np.max(self.chord_lengths)
        self.chord_lengths = np.asarray(self.chord_lengths) / max_len

        return self.chord_lengths
