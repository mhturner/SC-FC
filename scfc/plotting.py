import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

def addLinearFit(ax, x, y):
    r, p = pearsonr(x, y)
    coef = np.polyfit(x, y, 1)
    linfit = np.poly1d(coef)
    xx = np.linspace(x.min(), x.max(), 100)
    ax.plot(xx, linfit(xx), 'k-', LineWidth=2)
    print('r = {:.2f}'.format(r))


def addScaleBars(axis, dT, dF, T_value=-0.1, F_value=-0.4):
        axis.plot(T_value * np.ones((2)), np.array([F_value, F_value + dF]), 'k-', alpha=0.9)
        axis.plot(np.array([T_value, dT + T_value]), F_value * np.ones((2)), 'k-', alpha=0.9)
