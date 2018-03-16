import numpy as np
import matplotlib.pyplot as plt
import os
from bayesprob import log_evidence_for_optimization

savefig_path = '../FTR/figures/'
# plt.rc('text', usetex = True)


def grid_search(limits, X_train_orig, y_train, num_of_samples = 10):
    """
    Grid search over sigma2 and l.
    """
    sigma2_lo, sigma2_hi, l_lo, l_hi = limits
    sigma2_samples = np.linspace(sigma2_lo, sigma2_hi, num_of_samples)
    l_samples = np.linspace(l_hi, l_lo, num_of_samples)

    return np.array([[log_evidence_for_optimization([sigma2, l], X_train_orig, y_train)
                      for sigma2 in sigma2_samples] for l in l_samples])


def grid_visualize(data, limits):
    """
    Visualize the grid search using colormap.
    """
    cmap = plt.get_cmap('magma')
    img = plt.imshow(data, cmap = cmap, interpolation = 'nearest', extent = limits, aspect = 'auto')
    plt.colorbar(img, cmap = cmap)
    plt.xlabel('sigma2 (prior)')
    plt.ylabel('l (rbf)')
    limits_str = '-'.join(str(x) for x in limits)
    plt.savefig(os.path.join(savefig_path, 'grid-%s.png' % (limits_str)))
    plt.show()


