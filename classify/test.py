
import numpy
print numpy
import os

from sklearn import datasets


rawData = datasets.load_files("..//train_data")

for file in rawData['data']:
    print file
'''
X = rawData.data
print X[0] #first file content
y = rawData.target
print y
rawData = datasets.load_files("..//train_data", encoding="utf-8")
print unicode(rawData)
'''