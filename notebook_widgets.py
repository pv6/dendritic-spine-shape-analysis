import matplotlib.pyplot as plt
import numpy as np
from ipywidgets import interact, Layout
from CGAL.CGAL_Polyhedron_3 import Polyhedron_3
from typing import List, Tuple
from spine_segmentation import point_2_list, list_2_point, hash_point, \
    Segmentation
import meshplot as mp
import random


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
    @interact(x=(0, data.shape[1] - 1), y=(0, data.shape[0] - 1), z=(0, data.shape[2] - 1), layout=Layout(width='500px'))
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


def show_spines(spine_meshes: List[Polyhedron_3]):
    @interact(index=(0, len(spine_meshes)), correct_spine=True)
    def show_spine(index: int = 0, correct_spine: bool = True):
        print(index)
        show_3d_mesh(spine_meshes[index])
    return show_spine


def show_segmented_mesh(mesh: Polyhedron_3, segmentation: Segmentation):
    vertices, facets = _mesh_to_v_f(mesh)
    colors = np.ndarray((vertices.shape[0], 3))
    for i, vertex in enumerate(vertices):
        if segmentation[hash_point(list_2_point(vertex))]:
            colors[i] = RED
        else:
            colors[i] = GREEN
    mp.plot(vertices, facets, c=colors)

