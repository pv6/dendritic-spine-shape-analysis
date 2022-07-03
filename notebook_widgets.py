import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np
from ipywidgets import widgets
from CGAL.CGAL_Polyhedron_3 import Polyhedron_3
from CGAL.CGAL_Kernel import Vector_3, Point_3
from typing import List, Tuple, Dict, Set, Iterable
from spine_segmentation import point_2_list, list_2_point, hash_point, \
    Segmentation, segmentation_by_distance, local_threshold_3d,\
    spines_to_segmentation, correct_segmentation, get_spine_meshes, apply_scale
import meshplot as mp
from IPython.display import display
from spine_metrics import SpineMetric, SpineMetricDataset, calculate_metrics, \
    get_metric_class, HistogramSpineMetric, FloatSpineMetric, MeshDataset
from scipy.ndimage.measurements import label
from spine_clusterization import SpineClusterizer, KMeansSpineClusterizer, DBSCANSpineClusterizer
from pathlib import Path
import os
from sklearn.linear_model import LinearRegression


RED = (1, 0, 0)
GREEN = (0, 1, 0)
BLUE = (0, 0, 1)
WHITE = (1, 1, 1)
YELLOW = (1, 0.8, 0)
GRAY = (0.69, 0.69, 0.69)
DARK_GRAY = (0.30, 0.30, 0.30)
BLACK = (0.0, 0.0, 0.0)

V_F = Tuple[np.ndarray, np.ndarray]


class SpineMeshDataset:
    spine_meshes: MeshDataset
    dendrite_meshes: MeshDataset
    spine_to_dendrite: Dict[str, str]
    dendrite_to_spines: Dict[str, Set[str]]
    spine_v_f: Dict[str, V_F]
    dendrite_v_f: Dict[str, V_F]

    def __init__(self, spine_meshes: MeshDataset, dendrite_meshes: MeshDataset,
                 spine_to_dendrite: Dict[str, str]) -> None:
        # set fields
        self.spine_meshes = spine_meshes
        self.dendrite_meshes = dendrite_meshes
        self.spine_to_dendrite = spine_to_dendrite

        # generate dendrite to its spines mapping
        self.dendrite_to_spines = {name: set() for name in dendrite_meshes.keys()}
        for (spine_name, dendrite_name) in spine_to_dendrite.items():
            self.dendrite_to_spines[dendrite_name].add(spine_name)

        # calculate 'meshplot' mesh representations
        self._calculate_v_f()

    def get_dendrite_mesh(self, spine_name: str) -> Polyhedron_3:
        return self.dendrite_meshes[self.spine_to_dendrite[spine_name]]

    def get_dendrite_v_f(self, spine_name: str) -> V_F:
        return self.dendrite_v_f[self.spine_to_dendrite[spine_name]]

    def apply_scale(self, scale: Tuple[float, float, float]) -> None:
        def _apply_scale(mesh_dataset: MeshDataset) -> None:
            for (name, mesh) in mesh_dataset.items():
                mesh_dataset[name] = apply_scale(mesh, scale)
        _apply_scale(self.spine_meshes)
        _apply_scale(self.dendrite_meshes)
        self._calculate_v_f()

    def _calculate_v_f(self) -> None:
        self.spine_v_f = preprocess_meshes(self.spine_meshes)
        self.dendrite_v_f = preprocess_meshes(self.dendrite_meshes)


def _mesh_to_v_f(mesh: Polyhedron_3) -> V_F:
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
            if circulator == begin or j == 3:
                break
    return vertices, facets


def create_dir(dir_name: str) -> None:
    try:
        os.mkdir(dir_name)
    except OSError as error:
        pass


def load_spine_meshes(folder_path: str = "output",
                      spine_file_pattern: str = "**/spine_*.off") -> SpineMeshDataset:
    spine_meshes = {}
    dendrite_meshes = {}
    spine_to_dendrite = {}
    path = Path(folder_path)
    spine_names = list(path.glob(spine_file_pattern))
    for spine_name in spine_names:
        spine_meshes[str(spine_name)] = Polyhedron_3(str(spine_name))
        dendrite_path = str(spine_name.parent) + "\\surface_mesh.off"
        if dendrite_path not in dendrite_meshes:
            dendrite_meshes[dendrite_path] = Polyhedron_3(dendrite_path)
        spine_to_dendrite[str(spine_name)] = dendrite_path
    return SpineMeshDataset(spine_meshes, dendrite_meshes, spine_to_dendrite)


def preprocess_meshes(spine_meshes: MeshDataset) -> Dict[str, V_F]:
    output = {}
    for (spine_name, spine_mesh) in spine_meshes.items():
        output[spine_name] = _mesh_to_v_f(spine_mesh)
    return output


def show_3d_mesh(mesh: Polyhedron_3) -> None:
    vertices, facets = _mesh_to_v_f(mesh)
    mp.plot(vertices, facets)
    # mp.plot(vertices, facets, shading={"wireframe": True})


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
    # plot.add_lines(v[f[:, 0]], v[f[:, 1]], shading={"line_color": "gray"})
    plot.add_mesh(*_mesh_to_v_f(mesh), shading={"wireframe": True})


def _show_image(ax, image, mask=None, mask_opacity=0.5,
                cmap="gray", title=None):
    if mask is not None:
        indices = mask > 0
        mask = np.stack([np.zeros_like(mask), mask, np.zeros_like(mask)], -1)
        image = np.stack([image, image, image], -1)
        image[indices] = image[indices] * (1 - mask_opacity) + mask[indices] * mask_opacity

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


def _clusterization_to_colors(vertices: np.ndarray,
                              spine_meshes: MeshDataset,
                              clusterizer: SpineClusterizer,
                              dendrite_color: Tuple[float, float, float] = GREEN) -> np.ndarray:
    # generate segmentation for each cluster
    cluster_segmentations = []
    noise_cluster = clusterizer.noise_cluster
    for cluster in clusterizer.clusters:
        cluster_segmentations.append(spines_to_segmentation([spine_meshes[name] for name in cluster]))
    cluster_segmentations.append(spines_to_segmentation([spine_meshes[name] for name in noise_cluster]))

    # for each vertex check if belongs to cluster
    colors = np.ndarray((vertices.shape[0], 3))
    cluster_colors = clusterizer.grouping.colors
    cluster_colors.append(BLACK)
    for i, vertex in enumerate(vertices):
        hp = hash_point(list_2_point(vertex))
        colors[i] = dendrite_color
        for segmentation, color in zip(cluster_segmentations, cluster_colors):
            if hp in segmentation:
                colors[i] = color[:3]
                break
    return colors


class SpinePreview:
    widget: widgets.Widget
    spine_viewer: mp.Viewer
    dendrite_viewer: mp.Viewer

    spine_mesh: Polyhedron_3
    spine_name: str
    metrics: List[SpineMetric]

    spine_color: Tuple[float, float, float]

    _spine_colors: np.ndarray

    _dendrite_colors: np.ndarray

    _spine_v_f: V_F
    _dendrite_v_f: V_F

    _metrics_box: widgets.VBox

    _spine_mesh_id: int

    def __init__(self, spine_mesh: Polyhedron_3,
                 dendrite_v_f: V_F,
                 metrics: List[SpineMetric],
                 spine_name: str,
                 spine_color: Tuple[float, float, float] = RED) -> None:
        self.spine_color = spine_color
        self._spine_mesh_id = 0
        self._dendrite_v_f = dendrite_v_f
        self._set_spine_mesh(spine_mesh, metrics)
        self.spine_name = spine_name
        self.create_views()

    def create_views(self) -> None:
        preview_panel = widgets.HBox(children=[self._make_dendrite_view(),
                                               self._make_spine_panel()],
                                     layout=widgets.Layout(align_items="flex-start"))
        self.widget = widgets.VBox([widgets.Label(self.spine_name), preview_panel])

    def _set_spine_mesh(self, spine_mesh: Polyhedron_3, metrics: List[SpineMetric]) -> None:
        self.spine_mesh = spine_mesh
        self._spine_v_f = _mesh_to_v_f(self.spine_mesh)

        self._make_colors()

        if hasattr(self, "spine_viewer"):
            # self.spine_viewer.update_object(vertices=self._spine_v_f[0],
            #                                 faces=self._spine_v_f[1],
            #                                 colors=self._get_spine_colors())
            # self.spine_viewer.update_object(colors=self._get_spine_colors())
            self.spine_viewer.remove_object(self._spine_mesh_id)
            self._spine_mesh_id += 1
            self.spine_viewer.add_mesh(*self._spine_v_f, self._get_spine_colors())
        if hasattr(self, "dendrite_viewer"):
            self.dendrite_viewer.update_object(colors=self._get_dendrite_colors())

        self.metrics = metrics
        if hasattr(self, "_metrics_box"):
            self._fill_metrics_box()

    def _make_colors(self) -> None:
        # dendrite view colors
        self._dendrite_colors = np.ndarray((len(self._dendrite_v_f[0]), 3))
        self._dendrite_colors[:] = \
            _segmentation_to_colors(self._dendrite_v_f[0],
                                    spines_to_segmentation([self.spine_mesh]),
                                    GRAY, self.spine_color)
        # spine view colors
        self._spine_colors = np.ndarray((self.spine_mesh.size_of_vertices(), 3))
        self._spine_colors[:] = YELLOW

    def _get_dendrite_colors(self):
        return self._dendrite_colors

    def _get_spine_colors(self):
        return self._spine_colors

    def _make_dendrite_view(self) -> widgets.Widget:
        # make mesh viewer
        self.dendrite_viewer = make_viewer(*self._dendrite_v_f,
                                           self._get_dendrite_colors(), 400, 600)

        # set layout
        self.dendrite_viewer._renderer.layout = widgets.Layout(border="solid 1px")

        # title
        title = widgets.Label("Full View")

        return widgets.VBox(children=[title, self.dendrite_viewer._renderer])

    def _make_spine_view(self) -> widgets.Widget:
        # make mesh viewer
        self.spine_viewer = make_viewer(*self._spine_v_f,
                                        self._get_spine_colors(), 200, 200)

        # set layout
        self.spine_viewer._renderer.layout = widgets.Layout(border="solid 1px")

        # title
        title = widgets.Label("Spine View")

        return widgets.VBox(children=[title, self.spine_viewer._renderer])

    def _fill_metrics_box(self) -> None:
        self._metrics_box.children = [widgets.VBox([widgets.Label(metric.name),
                                                    metric.show()],
                                                   layout=widgets.Layout(border="solid 1px"))
                                      for metric in self.metrics]

    def _make_metrics_panel(self) -> widgets.Widget:
        # TODO: figure out scrolling
        self._metrics_box = widgets.VBox([], layout=widgets.Layout())
        self._fill_metrics_box()

        return widgets.VBox(children=[widgets.Label("Metrics"), self._metrics_box])

    def _make_spine_panel(self) -> widgets.Widget:
        # convert spine mesh to meshplot format
        return widgets.VBox(children=[self._make_spine_view(),
                                      self._make_metrics_panel()],
                            layout=widgets.Layout(align_items="flex-start"))


class SelectableSpinePreview(SpinePreview):
    is_selected_checkbox: widgets.Checkbox
    is_selected: bool

    _unselected_spine_colors: np.ndarray
    _unselected_dendrite_colors: np.ndarray

    metrics: List[SpineMetric]
    _metric_names: List[str]
    _metric_params: List[Dict]

    _correction_slider: widgets.IntSlider

    _dendrite_mesh: Polyhedron_3

    _initial_segmentation: Segmentation

    def __init__(self, spine_mesh: Polyhedron_3,
                 dendrite_v_f: V_F,
                 dendrite_mesh: Polyhedron_3,
                 metric_names: List[str],
                 metric_params: List[Dict] = None) -> None:
        self._initial_segmentation = spines_to_segmentation([spine_mesh])
        self.is_selected = True
        self._make_is_selected()
        self._dendrite_mesh = dendrite_mesh
        self._make_correction_slider()
        self._metric_names = metric_names
        self._metric_params = metric_params
        self._metrics = calculate_metrics(spine_mesh, self._metric_names,
                                          self._metric_params)
        super().__init__(spine_mesh, dendrite_v_f, self._metrics, "")

    def create_views(self) -> None:
        super().create_views()
        self.widget = widgets.VBox([self.is_selected_checkbox,
                                    self._correction_slider,
                                    self.widget])

    def _make_correction_slider(self) -> None:
        def observe_correction(change: Dict) -> None:
            if change["name"] == "value":
                self._correct_spine(change["new"])
        self._correction_slider = widgets.IntSlider(min=-6, max=6, value=0,
                                                    continuous_update=False,
                                                    description="Correction")
        self._correction_slider.observe(observe_correction)

    def _correct_spine(self, correction_value: int) -> None:
        new_segm = correct_segmentation(self._initial_segmentation,
                                        self._dendrite_mesh, correction_value)
        meshes = get_spine_meshes(self._dendrite_mesh, new_segm)
        # TODO: handle spine-splitting through correction slider
        if len(meshes) != 1:
            print(f"Oops, split this spine into {len(meshes)} spines.")
        self._set_spine_mesh(meshes[0], calculate_metrics(meshes[0],
                                                          self._metric_names,
                                                          self._metric_params))

    def _make_is_selected(self) -> None:
        def update_is_selected(change: Dict) -> None:
            if change["name"] == "value":
                self.set_selected(change["new"])
        self.is_selected_checkbox = widgets.Checkbox(value=self.is_selected,
                                                     description="Valid spine")
        self.is_selected_checkbox.observe(update_is_selected)

    def _make_colors(self) -> None:
        super()._make_colors()
        # dendrite view colors
        self._unselected_dendrite_colors = np.ndarray(
            (len(self._dendrite_v_f[0]), 3))
        self._unselected_dendrite_colors[:] = \
            _segmentation_to_colors(self._dendrite_v_f[0],
                                    spines_to_segmentation([self.spine_mesh]),
                                    GRAY, DARK_GRAY)
        # spine view colors
        self._unselected_spine_colors = np.ndarray(
            (self.spine_mesh.size_of_vertices(), 3))
        self._unselected_spine_colors[:] = GRAY

    def _get_dendrite_colors(self):
        if self.is_selected:
            return self._dendrite_colors
        return self._unselected_dendrite_colors

    def _get_spine_colors(self):
        if self.is_selected:
            return self._spine_colors
        return self._unselected_spine_colors

    def set_selected(self, value: bool) -> None:
        self.is_selected = value
        self.spine_viewer.update_object(self._spine_mesh_id, colors=self._get_spine_colors())
        self.dendrite_viewer.update_object(colors=self._get_dendrite_colors())


def _make_navigation_widget(slider: widgets.IntSlider, step=1) -> widgets.Widget:
    next_button = widgets.Button(description=">")
    prev_button = widgets.Button(description="<")

    def disable_buttons(change=None) -> None:
        next_button.disabled = slider.value >= slider.max
        prev_button.disabled = slider.value <= slider.min
        
    disable_buttons()
    slider.observe(disable_buttons)

    def next_callback(button: widgets.Button) -> None:
        slider.value += step
        disable_buttons()

    def prev_callback(button: widgets.Button) -> None:
        slider.value -= step
        disable_buttons()

    next_button.on_click(next_callback)
    prev_button.on_click(prev_callback)

    box = widgets.HBox([prev_button, next_button])
    return box
    

def select_spines_widget(spine_meshes: List[Polyhedron_3],
                         dendrite_mesh: Polyhedron_3,
                         metric_names: List[str],
                         metric_params: List[Dict] = None) -> widgets.Widget:
    dendrite_v_f: V_F = _mesh_to_v_f(dendrite_mesh)
    spine_previews = [SelectableSpinePreview(spine_mesh, dendrite_v_f,
                                             dendrite_mesh, metric_names,
                                             metric_params)
                      for spine_mesh in spine_meshes]

    def show_spine_by_index(index: int):
        # keeping old views caused bugs when switching between spines
        # this sacrifices saving camera position but oh well
        spine_previews[index].create_views()
        display(spine_previews[index].widget)

        # selected spine meshes and metrics
        return [(preview.spine_mesh, preview.metrics)
                for preview in spine_previews if preview.is_selected]

    slider = widgets.IntSlider(min=0, max=len(spine_meshes) - 1)
    navigation_buttons = _make_navigation_widget(slider)

    return widgets.VBox([navigation_buttons,
                         widgets.interactive(show_spine_by_index,
                                             index=slider)])


def interactive_segmentation(mesh: Polyhedron_3, correspondence,
                             reverse_correspondence,
                             skeleton_graph) -> widgets.Widget:
    vertices, facets = _mesh_to_v_f(mesh)

    slider = widgets.FloatLogSlider(min=-3.0, max=0.0, step=0.01, value=-1.0,
                                    continuous_update=False)
    correction_slider = widgets.IntSlider(min=-6, max=6, value=0,
                                          continuous_update=False)
    plot = mp.plot(vertices, facets)

    def do_segmentation(sensitivity=0.15, correction=0):
        segmentation = segmentation_by_distance(mesh, correspondence,
                                                reverse_correspondence,
                                                skeleton_graph, 1 - sensitivity)
        segmentation = correct_segmentation(segmentation, mesh, correction)
        plot.update_object(colors=_segmentation_to_colors(vertices, segmentation))

        return segmentation

    return widgets.interactive(do_segmentation, sensitivity=slider,
                               correction=correction_slider)


def show_segmented_mesh(mesh: Polyhedron_3, segmentation: Segmentation):
    vertices, facets = _mesh_to_v_f(mesh)
    colors = _segmentation_to_colors(vertices, segmentation)
    mp.plot(vertices, facets, c=colors)


def show_sliced_image(image: np.ndarray, x: int, y: int, z: int,
                      mask: np.ndarray = None, mask_opacity=0.5,
                      cmap="gray", title=""):
    fig, ax = plt.subplots(2, 2, figsize=(12, 10),
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

    mask_x = None
    mask_y = None
    mask_z = None
    if mask is not None:
        mask_x = mask[:, x, :]
        mask_y = mask[y, :, :].transpose()
        mask_z = mask[:, :, z]

    ax[0, 0].axis("off")
    _show_image(ax[1, 0], data_x, mask=mask_x, mask_opacity=mask_opacity, title=f"X = {x}", cmap=cmap)
    _show_image(ax[0, 1], data_y, mask=mask_y, mask_opacity=mask_opacity, title=f"Y = {y}", cmap=cmap)
    _show_image(ax[1, 1], data_z, mask=mask_z, mask_opacity=mask_opacity, title=f"Z = {z}", cmap=cmap)

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
        show_sliced_image(data, x, y, z, cmap=cmap)

    return display_slice


class Image3DRenderer:
    image: np.ndarray
    title: str

    _x: int
    _y: int
    _z: int

    def __init__(self, image: np.ndarray = np.zeros(0), title: str = "Title"):
        self._x = -1
        self._y = -1
        self._z = -1
        self.image = image
        self.title = title

    def show(self, cmap="gray"):
        shape = self.image.shape

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
            self._display_slice(x, y, z, cmap)

        return display_slice

    def _display_slice(self, x, y, z, cmap):
        show_sliced_image(self.image, x, y, z, cmap=cmap, title=self.title)


class MaskedImage3DRenderer(Image3DRenderer):
    mask: np.ndarray
    _mask_opacity: float

    def __init__(self, image: np.ndarray = np.zeros(0),
                 mask: np.ndarray = np.zeros(0), title: str = "Title"):
        super().__init__(image, title)
        self.mask = mask
        self._mask_opacity = 1

    def _display_slice(self, x, y, z, cmap):
        @widgets.interact(mask_opacity=widgets.FloatSlider(min=0, max=1,
                                                           value=self._mask_opacity,
                                                           step=0.1,
                                                           continuous_update=False))
        def display_slice_with_mask(mask_opacity):
            self._mask_opacity = mask_opacity
            show_sliced_image(self.image, x, y, z,
                              mask=self.mask, mask_opacity=mask_opacity,
                              cmap=cmap, title=self.title)


def interactive_binarization(image: np.ndarray) -> widgets.Widget:
    base_threshold_slider = widgets.IntSlider(min=0, max=255, value=127,
                                              continuous_update=False)
    weight_slider = widgets.IntSlider(min=0, max=100, value=5,
                                      continuous_update=False)
    block_size_slider = widgets.IntSlider(min=1, max=31, value=3, step=2,
                                          continuous_update=False)

    image_renderer = MaskedImage3DRenderer(title="Binarization Result")

    def show_binarization(base_threshold: int, weight: int, block_size: int) -> np.ndarray:
        result = local_threshold_3d(image, base_threshold=base_threshold,
                                    weight=weight / 100, block_size=block_size)
        image_renderer.image = image
        image_renderer.mask = result
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

    image_renderer = Image3DRenderer(title="Selected Connected Component")

    label_index_slider = widgets.IntSlider(min=0, max=len(used_labels) - 1,
                                           continuous_update=False)
    navigation_buttons = _make_navigation_widget(label_index_slider)
    display(navigation_buttons)

    def show_component(label_index: int) -> np.ndarray:
        lbl = used_labels[label_index]

        component = np.zeros_like(binary_image)
        component[labels == lbl] = 255

        preview = binary_image.copy()
        preview[preview > 0] = 64
        preview[labels == lbl] = 255

        image_renderer.image = preview
        image_renderer.show()

        return component

    return widgets.interactive(show_component, label_index=label_index_slider)


def clusterization_widget(clusterizer: SpineClusterizer,
                          spine_dataset: SpineMeshDataset,
                          metrics_dataset: SpineMetricDataset,
                          distance_metric=None) -> widgets.Widget:
    spine_previews_by_cluster = []
    colors = clusterizer.grouping.colors
    for index in range(clusterizer.num_of_clusters):
        spine_previews_by_cluster.append([])
        # cluster = clusterizer.get_representative_samples(index, 10, distance_metric)
        cluster = list(clusterizer.clusters[index])
        sample_number = min(4, len(cluster))
        cluster = cluster[:sample_number]
        for spine_name in cluster:
            spine_previews_by_cluster[-1].append(
                SpinePreview(spine_dataset.spine_meshes[spine_name],
                             spine_dataset.get_dendrite_v_f(spine_name),
                             metrics_dataset.row(spine_name),
                             spine_name, colors[index][:3]))

    def show_spine_by_cluster(cluster_index: int):
        def show_spine_by_index(index: int):
            # keeping old views caused bugs when switching between spines
            # this sacrifices saving camera position but oh well
            spine_previews_by_cluster[cluster_index][index].create_views()
            display(spine_previews_by_cluster[cluster_index][index].widget)
        slider = widgets.IntSlider(min=0, max=len(spine_previews_by_cluster[cluster_index]) - 1)
        navigation_buttons = _make_navigation_widget(slider)
        display(widgets.VBox([navigation_buttons,
                              widgets.interactive(show_spine_by_index,
                                                  index=slider)]))

    cluster_slider = widgets.IntSlider(min=0, max=max(len(spine_previews_by_cluster) - 1, 0))
    cluster_navigation_buttons = _make_navigation_widget(cluster_slider)
    return widgets.VBox([cluster_navigation_buttons,
                         widgets.interactive(show_spine_by_cluster,
                                             cluster_index=cluster_slider)])


# def new_clusterization_widget(clusterizer: SpineClusterizer,
#                               spine_meshes: Dict[str, V_F]) -> widgets.Widget:
#     def show_grid_by_cluster_index(cluster_index: int):
#         # extract representative cluster meshes
#         cluster = [(clusterizer.get_spine_reduced_coord(spine_name),
#                     spine_meshes[spine_name])
#                    for spine_name in clusterizer.get_representative_samples(cluster_index, 9)]
#         # sort by y
#         cluster.sort(key=lambda x: x[0][1])
#         # separate into rows
#         grid_size = int(np.ceil(np.sqrt(len(cluster))))
#         rows = [cluster[i:i + grid_size] for i in range(0, len(cluster), grid_size)]
#
#         # make rendering grid
#         grid_widget = widgets.VBox(
#             [widgets.HBox([make_viewer(*spine_mesh, None, 100, 100)._renderer
#                            for (_, spine_mesh) in row])
#              for row in rows])
#
#         display(widgets.HBox([clusterizer.show({cluster_index}),
#                               grid_widget]))
#
#     cluster_slider = widgets.IntSlider(min=0, max=max(clusterizer.num_of_clusters - 1, 0))
#     cluster_navigation_buttons = _make_navigation_widget(cluster_slider)
#
#     return widgets.VBox([cluster_navigation_buttons,
#                          widgets.interactive(show_grid_by_cluster_index,
#                                              cluster_index=cluster_slider)])


def new_new_clusterization_widget(clusterizer: SpineClusterizer,
                                  spine_dataset: SpineMeshDataset) -> widgets.Widget:
    colors = {}

    def show_dendrite_by_name(dendrite_name: str):
        if dendrite_name not in colors:
            colors[dendrite_name] = _clusterization_to_colors(
                spine_dataset.dendrite_v_f[dendrite_name][0],
                spine_dataset.spine_meshes,
                clusterizer)
        display(make_viewer(*spine_dataset.dendrite_v_f[dendrite_name],
                            colors[dendrite_name])._renderer)

    dendrite_name_dropdown = widgets.Dropdown(
        options=list(spine_dataset.dendrite_meshes.keys()),
        description="Dendrite:"
    )

    return widgets.interactive(show_dendrite_by_name, dendrite_name=dendrite_name_dropdown)


# def representative_clusterization_widget(clusterizer: SpineClusterizer,
#                                          spine_meshes: Dict[str, V_F],
#                                          num_of_samples: int = 3) -> widgets.Widget:
#     # TODO: deal with dictionaries better than this PLEASE
#     spine_mesh_list = list(spine_meshes.values())
#
#     # for each cluster
#     rows = []
#     for index in range(clusterizer.num_of_clusters):
#         row = [widgets.Label(f"{index}:")]
#         row.extend([make_viewer(*spine_mesh_list[i], None, 100, 100)._renderer
#                     for i in clusterizer.get_representative_samples(index, num_of_samples)])
#         rows.append(widgets.HBox(row))
#     return widgets.HBox([clusterizer.show(), widgets.VBox(rows)])


def clustering_experiment_widget(spine_metrics: SpineMetricDataset,
                                 every_spine_metrics: SpineMetricDataset,
                                 spine_dataset: SpineMeshDataset,
                                 clusterizer_type,
                                 param_slider_type, param_name,
                                 param_min_value, param_max_value, param_step,
                                 static_params: Dict,
                                 use_pca: bool = True) -> widgets.Widget:
    # calculate score graph
    pca_dim = 2 if use_pca else -1

    scores = {pca_dim: []}

    # scores = {-1: [], 2: []}
    # pca_dim = spine_metrics.row_as_array(spine_metrics.spine_names[0]).size // 2
    # while pca_dim >= 2:
    #     scores[pca_dim] = []
    #     pca_dim //= 2

    num_of_steps = int(np.ceil((param_max_value - param_min_value) / param_step))
    param_values = [np.clip(param_min_value + param_step * i,
                    param_min_value, param_max_value) for i in range(num_of_steps)]

    for (dim, dim_scores) in scores.items():
        for value in param_values:
            clusterizer = clusterizer_type(**{param_name: value}, **static_params, pca_dim=dim)
            clusterizer.fit(spine_metrics)
            # dim_scores.append(clusterizer.score())
            dim_scores.append(0)

    peak = np.nanargmax(scores[pca_dim])

    # reg = LinearRegression().fit(np.reshape(param_values, (-1, 1)), scores[2])

    param_slider = param_slider_type(min=param_min_value,
                                     max=param_max_value,
                                     value=param_values[peak],
                                     step=param_step,
                                     continuous_update=False)

    def show_clusterization(param_value) -> None:
        clusterizer = clusterizer_type(**{param_name: param_value}, **static_params, pca_dim=pca_dim)
        # clusterizer = clusterizer_type(**{param_name: param_value}, **static_params)
        clusterizer.fit(spine_metrics)

        score_graph = widgets.Output()
        with score_graph:
            plt.axvline(x=param_value, color='g', linestyle='-')
            plt.axhline(y=0, color='r', linestyle='-')
            for (dim, dim_scores) in scores.items():
                if dim == -1:
                    plot_label = "no pca"
                else:
                    plot_label = f"{dim}d with pca "
                plt.plot(param_values, dim_scores, label=plot_label)

            # plt.plot(param_values, reg.predict([[param] for param in param_values]))

            plt.title(clusterizer_type.__name__)
            plt.xlabel(param_name)
            plt.ylabel("Silhouette score")
            # plt.ylim([-1, 1])
            plt.legend(loc="lower right")
            # plt.rcParams["figure.figsize"] = (10, 10)
            plt.show()

        # display(widgets.VBox([widgets.HBox([clusterizer.show(), score_graph]),
        #                       new_clusterization_widget(clusterizer, spine_meshes)]))
        # display(widgets.VBox([widgets.HBox([clusterizer.show(), score_graph])]))
        display(widgets.VBox([widgets.HBox([clusterizer.show(spine_metrics), score_graph]),
                              clusterization_widget(clusterizer, spine_dataset, spine_metrics),
                              # new_new_clusterization_widget(clusterizer, spine_dataset),
                              # new_clusterization_widget(clusterizer, spine_dataset.spine_v_f),
                              # representative_clusterization_widget(clusterizer, spine_v_f, 5),
                              cluster_metric_distribution_widget(clusterizer, every_spine_metrics)
                              ]))
        # display(widgets.VBox([widgets.HBox([widgets.VBox([clusterizer.show(), clusterizer_pca.show()]), score_graph])]))

    clusterization_result = widgets.interactive(show_clusterization,
                                                param_value=param_slider)

    navigation_buttons = _make_navigation_widget(param_slider, param_step)

    return widgets.VBox([navigation_buttons, clusterization_result])


def k_means_clustering_experiment_widget(spine_metrics: SpineMetricDataset,
                                         every_spine_metrics: SpineMetricDataset,
                                         spine_dataset: SpineMeshDataset,
                                         min_num_of_clusters: int = 2,
                                         max_num_of_clusters: int = 20,
                                         metric="euclidean",
                                         use_pca: bool = True) -> widgets.Widget:
    return clustering_experiment_widget(spine_metrics, every_spine_metrics,
                                        spine_dataset,
                                        KMeansSpineClusterizer,
                                        widgets.IntSlider, "num_of_clusters",
                                        min_num_of_clusters, max_num_of_clusters,
                                        1, {"metric": metric}, use_pca)


def dbscan_clustering_experiment_widget(spine_metrics: SpineMetricDataset,
                                        every_spine_metrics: SpineMetricDataset,
                                        spine_dataset: SpineMeshDataset,
                                        metric="euclidean",
                                        min_eps: float = 2,
                                        max_eps: float = 20,
                                        eps_step: float = 0.1,
                                        use_pca: bool = True) -> widgets.Widget:
    return clustering_experiment_widget(spine_metrics, every_spine_metrics,
                                        spine_dataset,
                                        DBSCANSpineClusterizer,
                                        widgets.FloatSlider, "eps",
                                        min_eps, max_eps, eps_step,
                                        {"metric": metric}, use_pca)


def cluster_metric_distribution_widget(clusterizer: SpineClusterizer,
                                       metrics: SpineMetricDataset) -> widgets.Widget:
    metric_distributions = []
    for metric in metrics.row(list(metrics.spine_names)[0]):
        distribution_graph = widgets.Output()
        with distribution_graph:
            data = []
            colors = clusterizer.grouping.colors
            for i, cluster in enumerate(clusterizer.clusters):
                cluster_metrics = metrics.get_spines_subset(cluster)
                metric_column = cluster_metrics.column(metric.name)
                data.append(metric.get_distribution(metric_column))
                if issubclass(metric.__class__, HistogramSpineMetric):
                    value = metric.get_distribution(metric_column)
                    left_edges = [1.0 * i + j / len(value) for j in range(len(value))]
                    width = 1.0 * (left_edges[1] - left_edges[0])
                    plt.bar(left_edges, value, align='edge', width=width, color=colors[i])
            if issubclass(metric.__class__, FloatSpineMetric):
                plt.boxplot(data)
            plt.title(metric.name)
            plt.show()
        metric_distributions.append(distribution_graph)

    slider = widgets.IntSlider(min=0, max=max(0, len(metric_distributions) - 1))

    # navigation_buttons = _make_navigation_widget(slider)
    # return widgets.VBox([navigation_buttons,
    #                      widgets.interactive(lambda index: display(metric_distributions[index]),
    #                                          index=slider)])

    return widgets.HBox(metric_distributions, layout=widgets.Layout(width='3000px'))


def widget() -> widgets.Widget:
    pass