import numpy as np
from plotutil import *
from bayesprob import *


def print_info(X_train, y_train, X_test, y_test, w, mode = 'MLE', S_map = None, TYPE = 'untuned', sigma2 = None, l = None):
    fin = open('../FTR/tabs/cm-%s.tex' % (TYPE), 'r')
    buffer = fin.read()
    fin.close()

    fout = open('../FTR/tabs/cm-%s.tex' % (TYPE), 'w')
    buffer = buffer.replace('SIGMA2', '%g' % (sigma2))
    buffer = buffer.replace('STD', '%g' % (l))

    # Train/test log-likelihood
    ll_train = average_log_likelihood(w, y_train, X_train, S_map)
    ll_test = average_log_likelihood(w, y_test, X_test, S_map)
    print ('\n[%s]\n' % mode)
    print ('Train set log-likelihood:', ll_train)
    print ('Test set log-likelihood:', ll_test)
    buffer = buffer.replace('LL-TRAIN-%s' % (mode), '%.4f' % (ll_train))
    buffer = buffer.replace('LL-TEST-%s' % (mode), '%.4f' % (ll_test))

    # evaluate accuracy of our model
    y_test_hat = predict(X_test, w, S_map) > 0.5
    y_merged = y_test + y_test_hat
    true_neg_frac = np.count_nonzero(y_merged == 0) / np.count_nonzero(y_test == 0)
    true_pos_frac = np.count_nonzero(y_merged == 2) / np.count_nonzero(y_test == 1)

    # print confusion table
    print ()
    print ('Confusion Table:')
    print ('   %6d%8d' % (0, 1))
    print ('0  %6.4f%8.4f' % (true_neg_frac, 1 - true_neg_frac))
    print ('1  %6.4f%8.4f' % (1 - true_pos_frac, true_pos_frac))
    buffer = buffer.replace('TN-%s' % (mode), '%.4f' % (true_neg_frac))
    buffer = buffer.replace('FP-%s' % (mode), '%.4f' % (1 - true_neg_frac))
    buffer = buffer.replace('FN-%s' % (mode), '%.4f' % (1 - true_pos_frac))
    buffer = buffer.replace('TP-%s' % (mode), '%.4f' % (true_pos_frac))

    # print overall train/test accuracy
    y_train_hat = predict(X_train, w, S_map) > 0.5
    y_test_hat = predict(X_test, w, S_map) > 0.5
    train_accuracy = np.count_nonzero(y_train_hat - y_train == 0) / y_train.shape[0]
    test_accuracy = np.count_nonzero(y_test_hat - y_test == 0) / y_test.shape[0]
    print ()
    print ('Training accuracy = %.2f%%' % (train_accuracy * 100))
    print ('Test accuracy = %.2f%%' % (test_accuracy * 100))

    fout.write(buffer)
    fout.close()