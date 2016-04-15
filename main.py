#!/usr/bin/env python
# -- coding:utf-8 --

from classify import *
import numpy as np

if __name__ == '__main__':
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
        test_data = DataSet(test_folders)
        classifier = Classifier(train_data, test_data)
        classifier.train()
        precision[_k] = classifier.predict()
        print precision[_k]
    print np.mean(precision)
