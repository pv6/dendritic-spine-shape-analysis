import matplotlib.pyplot as plt
import numpy as np
from ipywidgets import widgets
from CGAL.CGAL_Polyhedron_3 import Polyhedron_3
from CGAL.CGAL_Kernel import Vector_3, Point_3
from typing import List, Tuple, Dict
from spine_segmentation import point_2_list, list_2_point, hash_point, \
    Segmentation, segmentation_by_distance, local_threshold_3d
import meshplot as mp
from IPython.display import display
from spine_metrics import SpineMetric


RED = (1, 0, 0)
GREEN = (0, 1, 0)
BLUE = (0, 0, 1)
WHITE = (1, 1, 1)
YELLOW = (1, 1, 0)
GRAY = (0.5, 0.5, 0.5)


def _mesh_to_v_f(mesh: Polyhedron_3) -> Tuple[np.ndarray, np.ndarray]:
    vertices = np.ndarray((mesh.size_of_vertices(), 3))
    for i, vertex in enumerate(mesh.vertices()):
        vertex.set_id(i)
        vertices[i, :] = point_2_list(vertex.point())

    facets = np.ndarray((mesh.size_of_facets(), 3)).astype("uint")
    for i, facet in enumerate(mesh.facets()):
        circulator = facet.facet_begin()
        j = 0
        begin = facet.facet_begin()
        while circulator.hasNext():
            halfedge = circulator.next()
            v = halfedge.vertex()
            facets[i, j] = (v.id())
            j += 1
            # check for end of loop
            if circulator == begin:
                break
    return vertices, facets


def show_3d_mesh(mesh: Polyhedron_3) -> None:
    vertices, facets = _mesh_to_v_f(mesh)
    mp.plot(vertices, facets)


def show_line_set(lines: List[Tuple[Point_3, Point_3]], mesh) -> None:
    # make vertices and facets
    vertices = np.ndarray((len(lines) * 2, 3))
    facets = np.ndarray((len(lines), 3)).astype("uint")
    for i, line in enumerate(lines):
        vertices[2 * i, :] = point_2_list(line[0])
        vertices[2 * i + 1, :] = point_2_list(line[1])

        facets[i, 0] = 2 * i
        facets[i, 1] = 2 * i + 1
        facets[i, 2] = 2 * i

    # render
    plot = mp.plot(vertices, facets, shading={"wireframe": True})
    v, f = _mesh_to_v_f(mesh)
    plot.add_lines(v[f[:, 0]], v[f[:, 1]], shading={"line_color": "gray"})
    # plot.add_mesh(*_mesh_to_v_f(mesh), shading={"wireframe": True})


def _show_image(ax, image, cmap="gray", title=None):
    ax.imshow(image, cmap=cmap)

    if title:
        ax.set_title(title)


def _show_cross_planes(ax, coord_1, coord_2, shape, color_1, color_2, border_color) -> None:
    # show plane 1
    ax.plot((coord_1, coord_1), (0, shape[0] - 1), color=color_1, lw=3)
    # show plane 2
    ax.plot((0, shape[1] - 1), (coord_2, coord_2), color=color_2, lw=3)
    # show border
    # horizontal
    ax.plot((0, shape[1] - 1), (0, 0), color=border_color, lw=3)
    ax.plot((0, shape[1] - 1), (shape[0] - 1, shape[0] - 1), color=border_color, lw=3)
    # vertical
    ax.plot((0, 0), (0, shape[0] - 1), color=border_color, lw=3)
    ax.plot((shape[1] - 1, shape[1] - 1), (0, shape[0] - 1), color=border_color, lw=3)


def make_viewer(v: np.ndarray, f: np.ndarray, c=None) -> mp.Viewer:
    view = mp.Viewer({})
    view.add_mesh(v, f, c)
    return view


class SpinePreview:
    widget: widgets.Widget
    view: mp.Viewer
    is_selected: widgets.Checkbox

    _selected_colors: np.ndarray
    _unselected_colors: np.ndarray

    def __init__(self, widget: widgets.Widget, view: mp.Viewer,
                 is_selected: widgets.Checkbox, size_of_v: int) -> None:
        self.widget = widget
        self.view = view
        self.is_selected = is_selected
        self._selected_colors = np.ndarray((size_of_v, 3))
        self._selected_colors[:] = YELLOW
        self._unselected_colors = np.ndarray((size_of_v, 3))
        self._unselected_colors[:] = GRAY

    def set_selected(self, value: bool) -> None:
        self.view.update_object(colors=self._selected_colors if value else self._unselected_colors)


def _get_spine_preview_widget(spine_v_f: Tuple, dendrite_v_f: Tuple,
                              metrics: List[SpineMetric]) -> SpinePreview:
    view = make_viewer(*spine_v_f)
    # view.add_mesh(*dendrite_v_f, shading={"wireframe": True})
    # (v, f) = dendrite_v_f
    # view.add_lines(v[f[:, 0]], v[f[:, 1]], shading={"line_color": "gray"})

    metrics_box = widgets.VBox([widgets.HBox([widgets.Label(metric.name),
                                              metric.show()])
                                for metric in metrics])

    view_and_metrics = widgets.HBox([view._renderer, metrics_box])
    is_selected = widgets.Checkbox(value=True)

    return SpinePreview(widgets.VBox([is_selected, view_and_metrics]), view,
                        is_selected, len(spine_v_f[0]))


def select_spines_widget(spine_meshes: List[Polyhedron_3],
                         dendrite_mesh: Polyhedron_3,
                         metrics: List[List[SpineMetric]]) -> widgets.Widget:

    dendrite_v_f: Tuple = _mesh_to_v_f(dendrite_mesh)
    spine_previews = [_get_spine_preview_widget(_mesh_to_v_f(spine_mesh),
                                                dendrite_v_f, metrics[i])
                      for i, spine_mesh in enumerate(spine_meshes)]

    spine_selection = [True for _ in range(len(spine_meshes))]
    for i, preview in enumerate(spine_previews):
        # set callback for checkbox value change
        # (capture i value via argument default value)
        def update_spine_selection(change: Dict, i=i) -> None:
            if change["name"] == "value":
                value = change["new"]
                spine_selection[i] = value
                spine_previews[i].set_selected(value)
        preview.is_selected.observe(update_spine_selection)

    def show_indexed_spine(index: int):
        print(index)
        display(spine_previews[index].widget)

        # return reference to         
        return spine_selection

    slider = widgets.IntSlider(min=0, max=len(spine_meshes) - 1)
    
    return widgets.interactive(show_indexed_spine, index=slider)


def _segmentation_to_colors(vertices: np.ndarray,
                            segmentation: Segmentation) -> np.ndarray:
    colors = np.ndarray((vertices.shape[0], 3))
    for i, vertex in enumerate(vertices):
        if hash_point(list_2_point(vertex)) in segmentation:
            colors[i] = RED
        else:
            colors[i] = GREEN
    return colors


def interactive_segmentation(mesh: Polyhedron_3, correspondence,
                             reverse_correspondence,
                             skeleton_graph) -> widgets.Widget:
    vertices, facets = _mesh_to_v_f(mesh)
    slider = widgets.FloatSlider(min=0.0, max=1.0, step=0.01, value=0.75,
                                 continuous_update=False)
    plot = mp.plot(vertices, facets)

    def do_segmentation(sensitivity=0.75):
        segmentation = segmentation_by_distance(mesh, correspondence,
                                                reverse_correspondence,
                                                skeleton_graph, sensitivity)
        plot.update_object(colors=_segmentation_to_colors(vertices, segmentation))

        return segmentation

    return widgets.interactive(do_segmentation, sensitivity=slider)


def show_segmented_mesh(mesh: Polyhedron_3, segmentation: Segmentation):
    vertices, facets = _mesh_to_v_f(mesh)
    colors = _segmentation_to_colors(vertices, segmentation)
    mp.plot(vertices, facets, c=colors)


def show_sliced_image(image, x, y, z, cmap="gray", title=""):
    fig, ax = plt.subplots(2, 2, figsize=(15, 5),
                           gridspec_kw={
                               'width_ratios': [image.shape[2],
                                                image.shape[1]],
                               'height_ratios': [image.shape[2],
                                                 image.shape[0]]
                           })

    if title != "":
        fig.suptitle(title)

    data_x = image[:, x, :]
    data_y = image[y, :, :].transpose()
    data_z = image[:, :, z]

    ax[0, 0].axis("off")
    _show_image(ax[1, 0], data_x, title=f"X = {x}", cmap=cmap)
    _show_image(ax[0, 1], data_y, title=f"Y = {y}", cmap=cmap)
    _show_image(ax[1, 1], data_z, title=f"Z = {z}", cmap=cmap)

    _show_cross_planes(ax[1, 0], z, y, data_x.shape, "blue", "green", "red")
    _show_cross_planes(ax[0, 1], x, z, data_y.shape, "red", "blue", "green")
    _show_cross_planes(ax[1, 1], x, y, data_z.shape, "red", "green", "blue")

    plt.tight_layout()
    plt.show()


def show_3d_image(data: np.ndarray, cmap="gray"):
    @widgets.interact(
        x=widgets.IntSlider(min=0, max=data.shape[1] - 1, continuous_update=False),
        y=widgets.IntSlider(min=0, max=data.shape[0] - 1, continuous_update=False),
        z=widgets.IntSlider(min=0, max=data.shape[2] - 1, continuous_update=False),
        layout=widgets.Layout(width='500px'))
    def display_slice(x, y, z):
        show_sliced_image(data, x, y, z, cmap)

    return display_slice


class Image3DRenderer:
    _x: int
    _y: int
    _z: int

    _images: List[Tuple[np.ndarray, str]]

    def __init__(self):
        self._x = -1
        self._y = -1
        self._z = -1
        self._images: List[Tuple[np.ndarray, str]] = []

    @property
    def images(self) -> List[Tuple[np.ndarray, str]]:
        return self._images.copy()

    @images.setter
    def images(self, value: List[Tuple[np.ndarray, str]]) -> None:
        self._images = value.copy()

    def show(self, cmap="gray"):
        shape = self._images[0][0].shape

        if self._x < 0:
            self._x = shape[1] // 2
        if self._y < 0:
            self._y = shape[0] // 2
        if self._z < 0:
            self._z = shape[2] // 2

        @widgets.interact(x=widgets.IntSlider(value=self._x, min=0,
                                              max=shape[1] - 1,
                                              continuous_update=False),
                          y=widgets.IntSlider(value=self._y, min=0,
                                              max=shape[0] - 1,
                                              continuous_update=False),
                          z=widgets.IntSlider(value=self._z, min=0,
                                              max=shape[2] - 1,
                                              continuous_update=False))
        def display_slice(x, y, z):
            self._x = x
            self._y = y
            self._z = z
            for i, (image, title) in enumerate(self._images):
                show_sliced_image(image, x, y, z, cmap=cmap, title=title)

        return display_slice


def interactive_binarization(image: np.ndarray) -> widgets.Widget:
    base_threshold_slider = widgets.IntSlider(min=0, max=255, value=127,
                                              continuous_update=False)
    weight_slider = widgets.IntSlider(min=0, max=100, value=5,
                                      continuous_update=False)
    block_size_slider = widgets.IntSlider(min=1, max=31, value=3, step=2,
                                          continuous_update=False)

    image_renderer = Image3DRenderer()

    def show_binarization(base_threshold: int, weight: int, block_size: int) -> np.ndarray:
        result = local_threshold_3d(image, base_threshold=base_threshold,
                                    weight=weight / 100, block_size=block_size)
        image_renderer.images = [(image, "Source Image"),
                                 (result, "Binarization Result")]
        image_renderer.show()

        return result

    return widgets.interactive(show_binarization,
                               base_threshold=base_threshold_slider,
                               weight=weight_slider,
                               block_size=block_size_slider)
