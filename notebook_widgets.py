import matplotlib.pyplot as plt
import numpy as np
from ipywidgets import widgets
from CGAL.CGAL_Polyhedron_3 import Polyhedron_3
from typing import List, Tuple, Dict
from spine_segmentation import point_2_list, list_2_point, hash_point, \
    Segmentation, segmentation_by_distance, load_segmentation
import meshplot as mp
from IPython.display import display
from spine_metrics import SpineMetric


RED = (1, 0, 0)
GREEN = (0, 1, 0)
BLUE = (0, 0, 1)
WHITE = (1, 1, 1)


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


def show_3d_image(data: np.ndarray, cmap="gray"):
    @widgets.interact(
        x=widgets.IntSlider(min=0, max=data.shape[1] - 1, continuous_update=False),
        y=widgets.IntSlider(min=0, max=data.shape[0] - 1, continuous_update=False),
        z=widgets.IntSlider(min=0, max=data.shape[2] - 1, continuous_update=False),
        layout=widgets.Layout(width='500px'))
    def display_slice(x, y, z):
        fig, ax = plt.subplots(2, 2, figsize=(15, 5),
                               gridspec_kw={
                                   'width_ratios': [data.shape[2],
                                                    data.shape[1]],
                                   'height_ratios': [data.shape[2],
                                                     data.shape[0]]
                               })

        data_x = data[:, x, :]
        data_y = data[y, :, :].transpose()
        data_z = data[:, :, z]

        ax[0, 0].axis("off")
        _show_image(ax[1, 0], data_x, title=f"X = {x}", cmap=cmap)
        _show_image(ax[0, 1], data_y, title=f"Y = {y}", cmap=cmap)
        _show_image(ax[1, 1], data_z, title=f"Z = {z}", cmap=cmap)

        _show_cross_planes(ax[1, 0], z, y, data_x.shape, "blue", "green", "red")
        _show_cross_planes(ax[0, 1], x, z, data_y.shape, "red", "blue", "green")
        _show_cross_planes(ax[1, 1], x, y, data_z.shape, "red", "green", "blue")

        plt.tight_layout()

        plt.show()

    return display_slice


def make_viewer(v: np.ndarray, f: np.ndarray, c=None) -> mp.Viewer:
    view = mp.Viewer({})
    view.add_mesh(v, f, c)
    return view


def _get_spine_preview_widget(spine_v_f: Tuple, dendrite_v_f: Tuple,
                              metrics: List[SpineMetric]) -> widgets.VBox:
    view = make_viewer(*spine_v_f)
    # view.add_mesh(*dendrite_v_f, shading={"wireframe": True})
    # (v, f) = dendrite_v_f
    # view.add_lines(v[f[:, 0]], v[f[:, 1]], shading={"line_color": "gray"})

    metrics_box = widgets.VBox([widgets.HBox([widgets.Label(metric.name),
                                              metric.show()])
                                for metric in metrics])

    view_and_metrics = widgets.HBox([view._renderer, metrics_box])
    is_selected = widgets.Checkbox(value=True)

    return widgets.VBox([is_selected, view_and_metrics])


def select_spines_widget(spine_meshes: List[Polyhedron_3],
                         dendrite_mesh: Polyhedron_3,
                         metrics: List[List[SpineMetric]]) -> widgets.Widget:

    dendrite_v_f: Tuple = _mesh_to_v_f(dendrite_mesh)
    spine_previews = [_get_spine_preview_widget(_mesh_to_v_f(spine_mesh),
                                                dendrite_v_f, metrics[i])
                      for i, spine_mesh in enumerate(spine_meshes)]

    spine_selection = [True for _ in range(len(spine_meshes))]
    for i in range(len(spine_previews)):
        # set callback for checkbox value change
        # (capture i value as argument default value)
        def update_spine_selection(change: Dict, i=i) -> None:
            if change["name"] == "value":
                spine_selection[i] = change["new"]
        spine_previews[i].children[0].observe(update_spine_selection)

    def show_indexed_spine(index: int):
        print(index)
        display(spine_previews[index])

        # return reference to         
        return spine_selection

    slider = widgets.IntSlider(min=0, max=len(spine_meshes) - 1)
    
    return widgets.interactive(show_indexed_spine, index=slider)


def _segmentation_to_colors(vertices: np.ndarray,
                            segmentation: Segmentation) -> np.ndarray:
    colors = np.ndarray((vertices.shape[0], 3))
    for i, vertex in enumerate(vertices):
        if segmentation[hash_point(list_2_point(vertex))]:
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
    plot = mp.plot(vertices, facets, c=colors)


if __name__ == "__main__":
    # load mesh and segmentation
    mesh = Polyhedron_3("output/surface_mesh.off")
    segmentation = load_segmentation("output/segmentation.json")

    # extract spine meshes
    from spine_segmentation import get_spine_meshes
    spine_meshes = get_spine_meshes(mesh, segmentation)

    # calculate metrics for each spine
    from spine_metrics import make_metrics
    metric_names = ["Area", "Volume"]
    metrics = []
    for spine_mesh in spine_meshes:
        metrics.append(make_metrics(spine_mesh, metric_names))
    
    selection_widget = select_spines_widget(spine_meshes, mesh, metrics)
    display(selection_widget)
