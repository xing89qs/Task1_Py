#!/usr/bin/env python
# -- coding:utf-8 --

from dataset import DataSet
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np


class Classifier(object):
    def __init__(self, train_data, test_data):
        self.train_data = train_data
        self.test_data = test_data
        pass

    def train(self):
        self.fit_vectorizer = TfidfVectorizer()
        self.train_feature = self.fit_vectorizer.fit_transform(self.train_data.data_list)
        pass

    def predict(self):
        self.test_feature = self.fit_vectorizer.transform(self.test_data.data_list)
        clf = LinearSVC(C=1)
        clf.fit(self.train_feature, np.array(self.train_data.label))
        pred = clf.predict(self.test_feature)

        right = 0
        for i in xrange(500):
            if pred[i]==self.test_data.label[i]:
                right+=1
        print right
        pass


if __name__ == '__main__':
    train_data = DataSet("train_data_tokens")
    test_data = DataSet("test_data_tokens")
    classifier = Classifier(train_data, test_data)
    classifier.train()
    classifier.predict()

