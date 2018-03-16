import numpy as np
from scipy.optimize import fmin_bfgs
from plotutil import *


def average_log_likelihood(w, y, X, S_map = None):
    """
    Calculate the log-likelihood per data point.
    """
    if S_map is None:
        z = np.dot(X, w)
    else:
        z = predict_Bayesian_before_sigmoid(X, w, S_map)
    # notice that sigmoid(-x) = 1 - sigmoid(x)
    return np.mean(y * log_sigmoid_stable(z) + (1 - y) * log_sigmoid_stable(-z))


def log_likelihood(w, y, X, sign = 1.0):
    """
    Calculate the value of the log-likelihood.
    """
    z = np.dot(X, w)
    # notice that sigmoid(-x) = 1 - sigmoid(x)
    return sign * np.sum(y * log_sigmoid_stable(z) + (1 - y) * log_sigmoid_stable(-z))


def grad_log_likelihood(w, y, X, sign = 1.0):
    """
    Calculate the gradient of the log-likelihood.
    """
    return sign * np.dot(X.T, y - sigmoid(np.dot(X, w)))


def log_posterior(w, y, X, sigma2, sign = 1.0):
    """
    The logarithm of the posterior distribution.

    params:
    sigma2 - variance of the gaussian prior

    Note:
    To find MAP solution using fmin_bfgs(), set 'sign' to -1.0
    """
    return sign * (-0.5 * np.dot(w, w) / sigma2 + log_likelihood(w, y, X))


def grad_log_posterior(w, y, X, sigma2, sign = 1.0):
    """
    The gradient of log_posterior.
    """
    return sign * (-w / sigma2 + grad_log_likelihood(w, y, X))


def find_MAP_solution(y, X, sigma2, param_size):
    """
    Find the MAP solution w.r.t the log_posterior,
    including w_map and A_map.
    """
    w_initial = np.zeros(param_size)
    w_map = fmin_bfgs(log_posterior, w_initial, grad_log_posterior, args = (y, X, sigma2, -1.0))
    A_map = np.eye(param_size) / sigma2 + np.sum(grad_sigmoid(np.dot(w_map, x)) * np.outer(x, x) for x in X)
    return w_map, A_map


def find_MLE_solution(y, X, param_size):
    w_initial = np.zeros(param_size)
    w_mle = fmin_bfgs(log_likelihood, w_initial, grad_log_likelihood, args = (y, X, -1.0), gtol=0.01)
    return w_mle


def compute_log_evidence(y, X, sigma2, w_map, A_map):
    """
    Compute the logarithm of model evidence.
    """
    param_size = w_map.shape[0]
    sign, logdet = np.linalg.slogdet(A_map) # for numerical stability
    return -0.5 * np.dot(w_map, w_map) / sigma2 - 0.5 * param_size * np.log(sigma2) \
           -0.5 * logdet + log_likelihood(w_map, y, X)


counter = 0


def log_evidence_for_optimization(hparams, X_train_orig, y, sign = 1.0):
    """
    Model evidence calculation dedicated to hyperparameter optimization.

    params:
    hparams - hyperparameters, sigma2 and l
    X_train_orig - original data points (training data only)
    """
    global counter
    counter += 1
    print ('\nCounter = %d\n' % (counter))

    sigma2, l = hparams
    X = data_transform(X_train_orig, 'rbf', std = l, target = X_train_orig)
    w_map, A_map = find_MAP_solution(y, X, sigma2, X.shape[1])
    return sign * compute_log_evidence(y, X, sigma2, w_map, A_map)

