# coding=utf-8

import jieba
import jieba.posseg as pseg
import os
import sys
from sklearn import feature_extraction
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer

dataset = os.listdir("..//train_data")

for type in dataset:
    for text in os.listdir("..//train_data/"+type):
        file = open("..//train_data/"+type+"/"+text)
        content = file.read()
        corpus = [
            content,
        ]
        vectorizer=CountVectorizer()#该类会将文本中的词语转换为词频矩阵，矩阵元素a[i][j] 表示j词在i类文本下的词频
        transformer=TfidfTransformer()#该类会统计每个词语的tf-idf权值
        tfidf=transformer.fit_transform(vectorizer.fit_transform(corpus))#第一个fit_transform是计算tf-idf，第二个fit_transform是将文本转为词频矩阵
        word=vectorizer.get_feature_names()#获取词袋模型中的所有词语
        print content
        for x in word:
            print x
