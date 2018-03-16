import matplotlib.pyplot as plt
import numpy as np
import os

savefig_path = '../FTR/figures/'


def plot_data_internal(X, y):
    """
    Core internal function for visualising input data (X, y).

    params:
    X - Original data points
    y - ground-truth labels (0 or 1)

    returns:
    xx, yy - meshgrids in corresponding axes(used for contour plotting).
    """
    x_min, x_max = X[ : , 0 ].min() - 0.5, X[ : , 0 ].max() + 0.5
    y_min, y_max = X[ : , 1 ].min() - 0.5, X[ : , 1 ].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
    plt.figure(figsize = (10, 6))
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    ax = plt.gca() # get current axis
    ax.plot(X[y == 0, 0], X[y == 0, 1], 'ro', label = 'Class 1')
    ax.plot(X[y == 1, 0], X[y == 1, 1], 'bo', label = 'Class 2')
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.title('Plot data')
    plt.legend(loc = 'upper left', scatterpoints = 1, numpoints = 1)
    return xx, yy


def plot_data(X, y, filename = None):
    """
    Visualise input data (X, y).

    params:
    X - Original data points
    y - ground-truth labels (0 or 1)
    filename - name of the figure to be saved. If value is None(default),
               the figure will not be saved.
    """
    xx, yy = plot_data_internal(X, y)
    if filename is not None:
        plt.savefig(os.path.join(savefig_path, filename))
    plt.show()


def sigmoid(x):
    """
    Calculate the sigmoid function of x.

    params:
    x - input value or vector

    returns:
    x_sigmoid - the sigmoid(sigmoid) of x
    """
    return 1.0 / (1.0 + np.exp(-x))


def grad_sigmoid(x):
    """
    Calculate the derivative of sigmoid function.
    """
    sig_x = sigmoid(x)
    return sig_x * (1 - sig_x)


def log_sigmoid_stable_elem(Xw_elem):
    """
    Numerically-stable calculation of log(sigmoid(Xw_elem))
    """
    if Xw_elem < -700:
        return Xw_elem
    return np.log(sigmoid(Xw_elem))


def log_sigmoid_stable(Xw):
    """
    Numerically-stable calculation of log(sigmoid(Xw))

    params:
    Xw - the input vector, equals to np.dot(X, w)

    returns:
    log(sigmoid(Xw)), with stable calculation
    """
    return np.array(list(map(log_sigmoid_stable_elem, Xw)))


def compute_average_ll(X, y, w):
    """
    Calculate the mean value of the log-likelihood.

    params:
    X - Transformed data points
    y - ground-truth labels (0 or 1)
    w - current parameter values

    returns:
    mean_ll - log-likelihood per data point.
    """
    Xw = np.dot(X, w)
    # notice that sigmoid(-x) = 1 - sigmoid(x)
    return np.mean(y * log_sig_stable(Xw) + (1 - y) * log_sig_stable(-Xw))


def compute_ll(X, y, w):
    """
    Calculate the value of the log-likelihood.
    """
    Xw = np.dot(X, w)
    return np.sum(y * log_sig_stable(Xw) + (1 - y) * log_sig_stable(-Xw))


def plot_ll(ll_train, ll_test, filename = None):
    """
    Plot average likelihood versus num of iterations.

    params:
    ll_* - average likelihood versus num of iterations(train/test)
    filename - name of the figure to be saved. If value is None(default),
               the figure will not be saved.
    """
    plt.figure(figsize = (10, 6))
    ax = plt.gca()
    plt.xlim(0, len(ll_train) + 2)
    plt.ylim(min(ll_train) - 0.1, max(ll_train) + 0.1)
    ax.plot(np.arange(1, len(ll_train) + 1), ll_train, 'r-')
    ax.plot(np.arange(1, len(ll_test) + 1), ll_test, 'b-')
    plt.xlabel('Steps')
    plt.ylabel('Average log-likelihood')
    plt.title('Plot Average Log-likelihood Curve')
    plt.legend(['Training set', 'Test set'])
    if filename is not None:
        plt.savefig(os.path.join(savefig_path, filename))
    plt.show()


def expand_inputs(std, base, target):
    """
    Expand the original input using RBF(Gaussian).

    params:
    std - hyper-parameter for the width of the Gaussian basis functions
    base - location of the Gaussian basis functions (width = 2)
    target - points at which to evaluate the basis functions (width = 2)

    returns:
    expanded - expanded data of shape (base.shape[0], target.shape[0])
    """
    base2 = np.sum(base ** 2, axis = 1)
    target2 = np.sum(target ** 2, axis = 1)
    ones_base = np.ones(base.shape[0])
    ones_target = np.ones(target.shape[0])
    r2 = np.outer(base2, ones_target) + np.outer(ones_base, target2) - 2 * np.dot(base, target.T)
    return np.exp(-0.5 * r2 / (std**2))


def data_transform(X_orig, mode='linear', **kwargs):
    """
    Transform the original data to data used in BGA.

    params:
    X_orig - original data to be transformed
    mode - 'linear' or 'rbf'
    kwargs - input dictionary for rbf parameters(std, target)

    returns:
    X_trans - the transformed input data
    """
    if mode == 'rbf':
        X_trans = expand_inputs(kwargs['std'], X_orig, kwargs['target'])
    elif mode == 'linear':
        X_trans = X_orig
    else:
        raise ValueError('Unrecognised mode: %s' % mode)
    return np.concatenate((X_trans, np.ones((X_trans.shape[0], 1))), axis = 1)


def predict_direct(X, w):
    """
    Direct prediction based on X and w.

    params:
    X - input (transformed) data to predict
    w - the parameter vector

    returns:
    y_prob - the probability distribution based on current model
    """
    return sigmoid(np.dot(X, w))


def predict_Bayesian_before_sigmoid(X, w_map, S_map):
    """
    Bayesian Prediction based on w_map and S_map(no sigmoid).
    """
    num = np.dot(X, w_map)
    den = np.sqrt(1 + np.pi / 8 * np.array([np.dot(x, np.dot(S_map, x)) for x in X]))
    return num / den


def predict_Bayesian(X, w_map, S_map):
    """
    Bayesian Prediction based on w_map and S_map.
    """
    return sigmoid(predict_Bayesian_before_sigmoid(X, w_map, S_map))


def predict(X, w, S_map = None):
    """
    Prediction for new data set based on given parameters.
    """
    if S_map is None:
        return predict_direct(X, w)
    else:
        return predict_Bayesian(X, w, S_map)


def plot_predictive_distribution(X_orig, y, w, filename=None, **kwargs):
    """
    Plot prediction probability contours for the original data.

    params:
    X_orig - original input data points
    y - ground-truth labels (0 or 1)
    w - the parameter vector
    mode - 'direct' or 'Bayesian'
    kwargs - input dictionary for necessary parameters
    filename - name of the figure to be saved. If value is None(default),
               the figure will not be saved.
    """
    xx, yy = plot_data_internal(X_orig, y)
    ax = plt.gca()
    X_mesh = np.concatenate((xx.ravel().reshape((-1, 1)), yy.ravel().reshape((-1, 1))), axis = 1)
    X_mesh_trans = data_transform(X_mesh, 'rbf', **kwargs)
    Z = predict(X_mesh_trans, w, kwargs['S_map'])
    Z = Z.reshape(xx.shape)
    cs2 = ax.contour(xx, yy, Z, cmap = 'RdBu', linewidths = 2)
    plt.clabel(cs2, fmt = '%2.1f', colors = 'k', fontsize = 14)
    if filename is not None:
        plt.savefig(os.path.join(savefig_path, filename))
    plt.show()
