import numpy as np
import shutil as sh
from plotutil import *
from bayesprob import *
from geninfo import *
from gridsearch import *

X_orig = np.loadtxt('X.txt')
y_orig = np.loadtxt('y.txt')

# Permute data randomly (with fixed seed for convenience)
np.random.seed(1313131)
rand_indices = np.random.permutation(X_orig.shape[0])
X_orig = X_orig[rand_indices, : ]
y_orig = y_orig[rand_indices]

# Hyperparameters
TYPE = 'tuned'
if TYPE == 'untuned':
    sigma2 = 1  # Gaussian prior variance
    l = 0.1     # RBF length scale
else:
    sigma2 = 0.83
    l = 0.49

# Train/Test set split
trainset_frac = 0.8
trainset_size = int(X_orig.shape[0] * trainset_frac)
X_train_orig = X_orig[ : trainset_size , : ]
X_test_orig = X_orig[ trainset_size : , : ]
X_train = data_transform(X_train_orig, 'rbf', std = l, target = X_train_orig)
X_test = data_transform(X_test_orig, 'rbf', std = l, target = X_train_orig)
y_train = y_orig[ : trainset_size ]
y_test  = y_orig[ trainset_size : ]

# Optimal hyperparameters grid search
sigma2_lo, sigma2_hi = 0.1, 10.0
l_lo, l_hi = 0.1, 2.0
limits = [sigma2_lo, sigma2_hi, l_lo, l_hi]
limits_str = '_'.join(str(x) for x in limits)

try:
    grid_data = np.load('grid_data_%s.npy' % (limits_str))
except FileNotFoundError:
    print ('\nLimits not found. Start a new grid search...')
    grid_data = grid_search(limits, X_train_orig, y_train, 10)
    np.save('grid_data_' + limits_str, grid_data)
finally:
    grid_visualize(grid_data, limits)

assert True

# Parameter vector w - extra bias term at last entry
param_size = trainset_size + 1

# Compute MLE, MAP and Bayesian solutions
#w_mle        = find_MLE_solution(y_train, X_train, param_size)
w_map, A_map = find_MAP_solution(y_train, X_train, sigma2, param_size)
S_map = np.linalg.inv(A_map)

# show predictions for MLE and MAP respectively
kwargs_basic = {'std' : l, 'target' : X_train_orig}
#plot_predictive_distribution(X_orig, y_orig, w_mle, 'pd-mle-notune.png', **kwargs_basic, S_map = None)
plot_predictive_distribution(X_orig, y_orig, w_map, 'pd-map-%s.png' % (TYPE), **kwargs_basic, S_map = None)
plot_predictive_distribution(X_orig, y_orig, w_map, 'pd-bayesian-%s.png' % (TYPE), **kwargs_basic, S_map = S_map)

#print_info(X_train, y_train, X_test, y_test, w_mle, 'MLE')
sh.copyfile('../FTR/tabs/cm-template.tex', '../FTR/tabs/cm-%s.tex' % (TYPE))
print_info(X_train, y_train, X_test, y_test, w_map, 'MAP', None, TYPE, sigma2, l)
print_info(X_train, y_train, X_test, y_test, w_map, 'Bayesian', S_map, TYPE, sigma2, l)
