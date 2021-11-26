import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
from ipywidgets import interact, Layout


def _show_image(ax, image, cmap="gray", title=None):
    ax.imshow(image, cmap=cmap)
    # ax.axis("off")

    if title:
        ax.set_title(title)


def _show_cross_planes(ax, coord_1, coord_2, shape, color_1, color_2) -> None:
    ax.plot((coord_1, coord_1), np.linspace(0, shape[0] - 1, 2), color=color_1, lw=3)
    ax.plot(np.linspace(0, shape[1] - 1, 2), (coord_2, coord_2), color=color_2, lw=3)


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

        _show_image(ax[0, 0], np.ones((data.shape[2], data.shape[2])), cmap=cmap)
        ax[0, 0].axis("off")
        _show_image(ax[1, 0], data_x, title="X Plane", cmap=cmap)
        _show_image(ax[0, 1], data_y, title="Y Plane", cmap=cmap)
        _show_image(ax[1, 1], data_z, title="Z Plane", cmap=cmap)

        _show_cross_planes(ax[1, 0], z, y, data_x.shape, "blue", "green")
        _show_cross_planes(ax[0, 1], x, z, data_y.shape, "red", "blue")
        _show_cross_planes(ax[1, 1], x, y, data_z.shape, "red", "green")

        plt.tight_layout()

        plt.show()

    return display_slice
