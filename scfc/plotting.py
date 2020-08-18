import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

def addLinearFit(ax, x, y):
    r, p = pearsonr(x, y)
    coef = np.polyfit(x, y, 1)
    linfit = np.poly1d(coef)
    xx = np.linspace(x.min(), x.max(), 100)
    ax.plot(xx, linfit(xx), 'k-', linewidth=2)
    print('r = {:.2f}'.format(r))


def addScaleBars(axis, dT, dF, T_value=-0.1, F_value=-0.4):
        axis.plot(T_value * np.ones((2)), np.array([F_value, F_value + dF]), 'k-', alpha=0.9)
        axis.plot(np.array([T_value, dT + T_value]), F_value * np.ones((2)), 'k-', alpha=0.9)

def overlayImage(im, mask, alpha, colors=None, z=0):
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