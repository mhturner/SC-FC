import numpy as np
import matplotlib.pyplot as plt

def addLinearFit(ax, x, y):
    r, p = pearsonr(x, y)
    coef = np.polyfit(x, y, 1)
    linfit = np.poly1d(coef)
    xx = np.linspace(x.min(), x.max(), 100)
    ax.plot(xx, linfit(xx), 'k-', LineWidth=3)
    print('r = {:.2f}'.format(r))
