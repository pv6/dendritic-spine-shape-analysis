import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np
from ipywidgets import widgets
from CGAL.CGAL_Polyhedron_3 import Polyhedron_3
from CGAL.CGAL_Kernel import Vector_3, Point_3
from typing import List, Tuple, Dict
from spine_segmentation import point_2_list, list_2_point, hash_point, \
    Segmentation, segmentation_by_distance, local_threshold_3d,\
    spines_to_segmentation
import meshplot as mp
from IPython.display import display
from spine_metrics import SpineMetric
from scipy.ndimage.measurements import label


RED = (1, 0, 0)
GREEN = (0, 1, 0)
BLUE = (0, 0, 1)
WHITE = (1, 1, 1)
YELLOW = (1, 0.8, 0)
GRAY = (0.69, 0.69, 0.69)
DARK_GRAY = (0.30, 0.30, 0.30)


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
    ax.imshow(image, norm=Normalize(0, 255), cmap=cmap)

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


def make_viewer(v: np.ndarray, f: np.ndarray, c=None, width: int = 600,
                height: int = 600) -> mp.Viewer:
    view = mp.Viewer({"width": width, "height": height})
    view.add_mesh(v, f, c)
    return view


class SpinePreview:
    widget: widgets.Widget
    spine_viewer: mp.Viewer
    dendrite_viewer: mp.Viewer
    is_selected_checkbox: widgets.Checkbox

    _selected_spine_colors: np.ndarray
    _unselected_spine_colors: np.ndarray

    _selected_dendrite_colors: np.ndarray
    _unselected_dendrite_colors: np.ndarray

    def __init__(self, spine_mesh: Polyhedron_3,
                 dendrite_v_f: Tuple[np.ndarray, np.ndarray],
                 metrics: List[SpineMetric]) -> None:
        preview_panel = widgets.HBox(children=[self._make_dendrite_view(spine_mesh, dendrite_v_f),
                                               self._make_spine_panel(spine_mesh, metrics)],
                                     layout=widgets.Layout(align_items="flex-start"))

        is_selected_checkbox = widgets.Checkbox(value=True, description="Valid spine")

        self.widget = widgets.VBox([is_selected_checkbox, preview_panel])
        self.is_selected_checkbox = is_selected_checkbox

    def _make_dendrite_view(self, spine_mesh: Polyhedron_3, dendrite_v_f: Tuple) -> widgets.Widget:
        # generate colors
        self._selected_dendrite_colors = np.ndarray((len(dendrite_v_f[0]), 3))
        self._selected_dendrite_colors[:] = \
            _segmentation_to_colors(dendrite_v_f[0],
                                    spines_to_segmentation([spine_mesh]),
                                    GREEN, RED)
        self._unselected_dendrite_colors = np.ndarray((len(dendrite_v_f[0]), 3))
        self._unselected_dendrite_colors[:] = \
            _segmentation_to_colors(dendrite_v_f[0],
                                    spines_to_segmentation([spine_mesh]),
                                    GRAY, DARK_GRAY)

        # make mesh viewer
        self.dendrite_viewer = make_viewer(*dendrite_v_f, self._selected_dendrite_colors, 400, 600)

        # set layout
        self.dendrite_viewer._renderer.layout = widgets.Layout(border="solid 1px")

        # title
        title = widgets.Label("Full View")

        return widgets.VBox(children=[title, self.dendrite_viewer._renderer])

    def _make_spine_view(self, spine_v_f: Tuple) -> widgets.Widget:
        # generate colors
        self._selected_spine_colors = np.ndarray((len(spine_v_f[0]), 3))
        self._selected_spine_colors[:] = YELLOW
        self._unselected_spine_colors = np.ndarray((len(spine_v_f[0]), 3))
        self._unselected_spine_colors[:] = GRAY

        # make mesh viewer
        self.spine_viewer = make_viewer(*spine_v_f, self._selected_spine_colors, 200, 200)

        # set layout
        self.spine_viewer._renderer.layout = widgets.Layout(border="solid 1px")

        # title
        title = widgets.Label("Spine View")

        return widgets.VBox(children=[title, self.spine_viewer._renderer])

    def _make_metrics_panel(self, metrics: List[SpineMetric]) -> widgets.Widget:
        # TODO: figure out scrolling
        metrics_box = widgets.VBox([widgets.VBox([widgets.Label(metric.name),
                                                  metric.show()],
                                                 layout=widgets.Layout(border="solid 1px"))
                                    for metric in metrics],
                                   layout=widgets.Layout())

        return widgets.VBox(children=[widgets.Label("Metrics"), metrics_box])

    def _make_spine_panel(self, spine_mesh, metrics) -> widgets.Widget:
        # convert spine mesh to meshplot format
        spine_v_f = _mesh_to_v_f(spine_mesh)

        return widgets.VBox(children=[self._make_spine_view(spine_v_f),
                                      self._make_metrics_panel(metrics)],
                            layout=widgets.Layout(align_items="flex-start"))

    def set_selected(self, value: bool) -> None:
        self.spine_viewer.update_object(colors=self._selected_spine_colors if value else self._unselected_spine_colors)
        self.dendrite_viewer.update_object(colors=self._selected_dendrite_colors if value else self._unselected_dendrite_colors)


def select_spines_widget(spine_meshes: List[Polyhedron_3],
                         dendrite_mesh: Polyhedron_3,
                         metrics: List[List[SpineMetric]]) -> widgets.Widget:

    dendrite_v_f: Tuple = _mesh_to_v_f(dendrite_mesh)
    spine_previews = [SpinePreview(spine_mesh,
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
        preview.is_selected_checkbox.observe(update_spine_selection)

    def show_indexed_spine(index: int):
        display(spine_previews[index].widget)

        # return indices of selected spines
        return [i for i, is_selected in enumerate(spine_selection) if is_selected]

    slider = widgets.IntSlider(min=0, max=len(spine_meshes) - 1)
    
    return widgets.interactive(show_indexed_spine, index=slider)


def _segmentation_to_colors(vertices: np.ndarray,
                            segmentation: Segmentation,
                            dendrite_color: Tuple[float, float, float] = GREEN,
                            spine_color: Tuple[float, float, float] = RED) -> np.ndarray:
    colors = np.ndarray((vertices.shape[0], 3))
    for i, vertex in enumerate(vertices):
        if hash_point(list_2_point(vertex)) in segmentation:
            colors[i] = spine_color
        else:
            colors[i] = dendrite_color
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


def select_connected_component_widget(binary_image: np.ndarray) ->widgets.Widget:
    # find connected components
    labels, num_of_components = label(binary_image)

    # sort labels by size
    unique, counts = np.unique(labels, return_counts=True)
    unique = unique.tolist()
    counts = counts.tolist()
    unique.sort(key=lambda x: counts[x], reverse=True)
    counts.sort(reverse=True)

    # filter background and too small labels
    used_labels = []
    for i, count in enumerate(counts):
        if count >= 10 and unique[i] != 0:
            used_labels.append(unique[i])

    image_renderer = Image3DRenderer()

    label_index_slider = widgets.IntSlider(min=0, max=len(used_labels) - 1,
                                           continuous_update=False)

    def show_component(label_index: int) -> np.ndarray:
        lbl = used_labels[label_index]

        component = np.zeros_like(binary_image)
        component[labels == lbl] = 255

        preview = binary_image.copy()
        preview[preview > 0] = 64
        preview[labels == lbl] = 255

        image_renderer.images = [(preview, "Selected Connected Component")]
        image_renderer.show()

        return component

    return widgets.interactive(show_component, label_index=label_index_slider)
