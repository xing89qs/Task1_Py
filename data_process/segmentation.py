#!/usr/bin/env python
# -- coding:utf-8 --
# 分词类Segmentation

import jieba


class Segmentation(object):
    """分词类"""

    __STOP_WORDS = set()
    __FILE_ENCODING = 'UTF-8'

    def __new__(cls):
        if not Segmentation.__STOP_WORDS:
            Segmentation.__load_stop_words()
        return object.__new__(cls)

    @staticmethod
    def __load_stop_words():
        stop_words_file = open(u"..//stop.dic", 'r')
        for word in stop_words_file:
            word = word.strip()
            Segmentation.__STOP_WORDS.add(unicode(word, Segmentation.__FILE_ENCODING))

    def cut(self, text):
        token_list = jieba.cut(text)
        new_token = []
        for token in token_list:
            token = token.strip()
            if token is None:
                continue
            if token not in Segmentation.__STOP_WORDS:
                new_token.append(token)
        return new_token
