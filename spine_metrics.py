from CGAL.CGAL_Polyhedron_3 import Polyhedron_3
from CGAL.CGAL_Polygon_mesh_processing import area, volume
from typing import Any, List
import ipywidgets as widgets


class SpineMetric:
    name: str
    value: Any

    def __init__(self, spine_mesh: Polyhedron_3):
        self.value = self._calculate(spine_mesh)

    def _calculate(self, spine_mesh: Polyhedron_3) -> Any:
        pass

    def show(self):
        pass


def calculate_metrics(spine_mesh: Polyhedron_3,
                      metric_names: List[str]) -> List[SpineMetric]:
    out = []
    for name in metric_names:
        klass = globals()[name.capitalize() + "SpineMetric"]
        out.append(klass(spine_mesh))
    return out


class FloatSpineMetric(SpineMetric):
    def show(self):
        return widgets.Label(f"{self.value:.2f}")


class AreaSpineMetric(FloatSpineMetric):
    def __init__(self, spine_mesh: Polyhedron_3) -> None:
        super().__init__(spine_mesh)
        self.name = "Area"

    def _calculate(self, spine_mesh: Polyhedron_3) -> Any:
        return area(spine_mesh)


class VolumeSpineMetric(FloatSpineMetric):
    def __init__(self, spine_mesh: Polyhedron_3) -> None:
        super().__init__(spine_mesh)
        self.name = "Volume"

    def _calculate(self, spine_mesh: Polyhedron_3) -> Any:
        return volume(spine_mesh)
