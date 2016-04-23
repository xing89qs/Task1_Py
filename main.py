#!/usr/bin/env python
# -- coding:utf-8 --

from classify import *
import numpy as np
from feature import *
import sys
import matplotlib as plot

def run(feature_selection=None, num=5000):
    _K = 5
    precision = np.zeros(_K)
    for _k in xrange(_K):
        train_folders = []
        test_folders = []
        for _i in xrange(_K):
            if _i == _k:
                test_folders.append("text classification_tokens/" + str(_i))
            else:
                train_folders.append("text classification_tokens/" + str(_i))
        train_data = DataSet(train_folders)
        if feature_selection is not None:
            feature_selector = FEATURE_SELECTORS[feature_selection]()
            train_data = feature_selector.select(train_data, num=num)
        test_data = DataSet(test_folders)
        classifier = Classifier(train_data, test_data)
        classifier.train()
        precision[_k] = classifier.predict()
    print np.mean(precision)
    return np.mean(precision)


if __name__ == '__main__':
    # base-line
    # run()

    # Feature Selection


    _NUMS = range(1000,  10000, 1000)

    '''
    run(feature_selection='df')
    run(feature_selection='mi')
    run(feature_selection='ig')
    run(feature_selection='wllr')
    run(feature_selection='chi')
    run(feature_selection='bns')
    '''

    run(feature_selection='all')
