import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# connection between the 8 points of 3d bbox
BONES_3D_BBOX = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 0),
    (0, 4),
    (1, 5),
    (2, 6),
    (3, 7),
    (4, 5),
    (5, 6),
    (6, 7),
    (7, 4),
]


def plot_2d_bbox(bbox_2d, bones, color, ax):
    if ax is None:
        axx = plt
    else:
        axx = ax
    colors = cm.rainbow(np.linspace(0, 1, len(bbox_2d)))
    for pt, c in zip(bbox_2d, colors):
        axx.scatter(pt[0], pt[1], color=c, s=50)

    if bones is None:
        bones = BONES_3D_BBOX
    for bone in bones:
        sidx, eidx = bone
        # bottom of bbox is white
        if min(sidx, eidx) >= 4:
            color = "w"
        axx.plot(
            [bbox_2d[sidx][0], bbox_2d[eidx][0]],
            [bbox_2d[sidx][1], bbox_2d[eidx][1]],
            color,
        )
    return axx


# http://www.icare.univ-lille1.fr/tutorials/convert_a_matplotlib_figure
def fig2data(fig):
    """
    @brief Convert a Matplotlib figure to a 4D
    numpy array with RGBA channels and return it
    @param fig a matplotlib figure
    @return a numpy 3D array of RGBA values
    """
    # draw the renderer
    fig.canvas.draw()

    # Get the RGBA buffer from the figure
    w, h = fig.canvas.get_width_height()
    buf = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
    buf.shape = (w, h, 4)

    # canvas.tostring_argb give pixmap in ARGB mode.
    # Roll the ALPHA channel to have it in RGBA mode
    buf = np.roll(buf, 3, axis=2)
    return buf


# http://www.icare.univ-lille1.fr/tutorials/convert_a_matplotlib_figure
def fig2img(fig):
    """
    @brief Convert a Matplotlib figure to a PIL Image
    in RGBA format and return it
    @param fig a matplotlib figure
    @return a Python Imaging Library ( PIL ) image
    """
    # put the figure pixmap into a numpy array
    buf = fig2data(fig)
    w, h, _ = buf.shape
    return Image.frombytes("RGBA", (w, h), buf.tobytes())


def concat_pil_images(images):
    """
    Put a list of PIL images next to each other
    """
    assert isinstance(images, list)
    widths, heights = zip(*(i.size for i in images))

    total_width = sum(widths)
    max_height = max(heights)

    new_im = Image.new("RGB", (total_width, max_height))

    x_offset = 0
    for im in images:
        new_im.paste(im, (x_offset, 0))
        x_offset += im.size[0]
    return new_im


def stack_pil_images(images):
    """
    Stack a list of PIL images next to each other
    """
    assert isinstance(images, list)
    widths, heights = zip(*(i.size for i in images))

    total_height = sum(heights)
    max_width = max(widths)

    new_im = Image.new("RGB", (max_width, total_height))

    y_offset = 0
    for im in images:
        new_im.paste(im, (0, y_offset))
        y_offset += im.size[1]
    return new_im


def im_list_to_plt(image_list, figsize, title_list=None):
    fig, axes = plt.subplots(nrows=1, ncols=len(image_list), figsize=figsize)
    for idx, (ax, im) in enumerate(zip(axes, image_list)):
        ax.imshow(im)
        ax.set_title(title_list[idx])
    fig.tight_layout()
    im = fig2img(fig)
    plt.close()
    return im
