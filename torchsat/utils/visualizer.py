import matplotlib.pyplot as plt
import tifffile
import numpy as np

__all__ = ["plot_img", "plot_bbox", "plot_mask"]

color_map = {
    1: "tab:blue",
    2: "tab:orange",
    3: "tab:green",
    4: "tab:red",
    5: "tab:purple",
    6: "tab:brown",
    7: "tab:pink",
    8: "tab:olive",
    9: "tab:cyan",
}


def plot_img(img, channels=(1, 2, 3), **kwargs):
    """plot the ndarray images
    
    Args:
        img (ndarray): the ndarray image.
        channels (tuple, optional): target channel to show. Defaults to (1,2,3).
        kwargs (dict, optional): the plt.imshow kwargs.
    """
    plt.figure(figsize=(10, 8))
    channels = channels if isinstance(channels, int) else [x - 1 for x in channels]
    if img.ndim == 2:
        tifffile.imshow(img, cmap="gray", **kwargs)
    else:
        tifffile.imshow(img[..., channels], **kwargs)


def plot_bbox(img, bboxes, channels=(1, 2, 3), labels=None, classes=None, **kwargs):
    """plot the image with bboxes
    
    Args:
        img (ndarray): the ndarray image.
        bboxes (ndarray): the bboxes to show
        channels (tuple, optional): target channel to show. Defaults to (1,2,3).
        labels ([type], optional): the label id corresponding to the bbox . Defaults to None.
        classes ([type], optional): the text name corresponding to the label id. Defaults to None.
        kwargs (dict, optional): the plt.imshow kwargs.
    Examples:
        >>> import numpy as np
        >>> from PIL import Image
        >>> fp = './tests/fixtures/different-types/jpeg_3channel_uint8.jpeg'
        >>> img = np.array(Image.open(fp))
        >>> bboxes = np.array([[  0.,   2., 100., 100.],
                             [140., 300., 200., 350.]])
        >>> plot_bbox(img, bboxes, labels=[1,4], classes=['car', 'building', 'tree', 'road'])
    """
    plt.figure(figsize=(10, 8))
    channels = channels if isinstance(channels, int) else [x - 1 for x in channels]
    if img.ndim == 2:
        plt.imshow(img, cmap="gray", **kwargs)
    else:
        plt.imshow(img[..., channels], **kwargs)

    if labels is None:
        labels = [1] * len(bboxes)

    for bbox, label in zip(bboxes.tolist(), labels):
        rect = plt.Rectangle(
            xy=(bbox[0], bbox[1]),
            width=bbox[2] - bbox[0],
            height=bbox[3] - bbox[1],
            fill=False,
            edgecolor=color_map[label],
            linewidth=2,
        )
        plt.gca().add_patch(rect)
        if classes is not None:
            text_bbox = dict(
                fc=color_map[label], alpha=0.5, lw=0, boxstyle="Square, pad=0"
            )
            plt.gca().annotate(
                classes[label - 1], xy=(bbox[0], bbox[1]), bbox=text_bbox, color="w"
            )

    plt.show()


def plot_mask(img, mask, channels=(1, 2, 3), mask_alpha=0.6, **kwargs):
    """plot image with mask
    
    Args:
        img (ndarray): the input ndarray image.
        mask (ndarray): the mask to show.
        channels (tuple, optional): target channel to show.. Defaults to (1,2,3).
        mask_alpha (float, optional): mask alpha. Defaults to 0.6.
        kwargs (dict, optional): the plt.imshow kwargs.
    """
    plt.figure(figsize=(10, 8))
    channels = channels if isinstance(channels, int) else [x - 1 for x in channels]
    if img.ndim == 2:
        plt.imshow(img, cmap="gray", **kwargs)
    else:
        plt.imshow(img[..., channels], **kwargs)
    masked_data = np.ma.masked_where(mask == 0, mask)
    plt.imshow(masked_data, alpha=mask_alpha, cmap="viridis")
