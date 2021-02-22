"""
Turner, Mann, Clandinin: Plotting utils.

https://github.com/mhturner/SC-FC
mhturner@stanford.edu
"""
import numpy as np
from scipy.stats import pearsonr
import matplotlib.colors
import matplotlib.pyplot as plt


def addLinearFit(ax, x, y, alpha=1):
    """Add linear fit to xy scatter plot."""
    r, p = pearsonr(x, y)
    coef = np.polyfit(x, y, 1)
    linfit = np.poly1d(coef)
    xx = np.linspace(x.min(), x.max(), 100)
    ax.plot(xx, linfit(xx), 'k-', linewidth=2, alpha=alpha)
    return r, p


def addScaleBars(axis, dT, dF, T_value=-0.1, F_value=-0.4):
    """Add scale bars to plot or image."""
    axis.plot(T_value * np.ones((2)), np.array([F_value, F_value + dF]), 'k-', alpha=0.9)
    axis.plot(np.array([T_value, dT + T_value]), F_value * np.ones((2)), 'k-', alpha=0.9)


def categorical_cmap(nc, nsc, cmap="tab10", continuous=False):
    """
    Make categorical cmap with nc ategories and nsc subcategories.

    from https://stackoverflow.com/a/47232942
    """
    if nc > plt.get_cmap(cmap).N:
        raise ValueError("Too many categories for colormap.")
    if continuous:
        ccolors = plt.get_cmap(cmap)(np.linspace(0, 1, nc))
    else:
        ccolors = plt.get_cmap(cmap)(np.arange(nc, dtype=int))
    cols = np.zeros((nc*nsc, 3))
    for i, c in enumerate(ccolors):
        chsv = matplotlib.colors.rgb_to_hsv(c[:3])
        arhsv = np.tile(chsv, nsc).reshape(nsc, 3)
        arhsv[:, 1] = np.linspace(chsv[1], 0.25, nsc)
        arhsv[:, 2] = np.linspace(chsv[2], 1, nsc)
        rgb = matplotlib.colors.hsv_to_rgb(arhsv)
        cols[i*nsc:(i+1)*nsc, :] = rgb
    cmap = matplotlib.colors.ListedColormap(cols)
    return cmap


def overlayImage(im, mask, alpha, colors=None, z=0):
    """Overlay image and mask."""
    im = im / np.max(im)
    imRGB = np.tile(im[..., np.newaxis], 3)[:, :, z, :]

    overlayComponent = 0
    origImageComponent = 0
    compositeMask = np.tile(mask[0][:, :, z, np.newaxis], 3)
    for ind, currentRoi in enumerate(mask):
        maskRGB = np.tile(currentRoi[:, :, z, np.newaxis], 3)
        newColor = colors[ind][:3]

        compositeMask = compositeMask + maskRGB
        overlayComponent += alpha * np.array(newColor) * maskRGB
        origImageComponent += (1 - alpha) * maskRGB * imRGB

    untouched = (compositeMask == False) * imRGB

    im_out = untouched + overlayComponent + origImageComponent
    im_out = (im_out * 255).astype(np.uint8)
    return im_out
