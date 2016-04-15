#!/usr/bin/env python
# -- coding:utf-8 --

import os
from segmentation import Segmentation


def write_to_file(path, filename, data, label):
    if not os.path.exists(path):
        os.makedirs(path)
    file = open(path + "/" + filename, 'w')
    for token in data:
        token = token.encode('utf-8')
        file.write(token)
        file.write('\n')
    file.close()


_K = 5


def process_data(folder_name):
    for _k in xrange(_K):
        data_dir = os.listdir("..//" + folder_name + "/" + str(_k))
        for label in data_dir:
            for sample in os.listdir("..//" + folder_name + "/" + str(_k) + "/" + label):
                if not str(sample).endswith(".txt"):
                    continue
                file = open("..//" + folder_name + "/" + label + "/" + sample, 'r')
                content = file.read()
                file.close()
                segmentation = Segmentation()
                token_list = segmentation.cut(content)
                write_to_file("..//" + folder_name + "_tokens/" + str(_k) + "/" + label, sample, token_list,
                              label)


def process():
    process_data("text classification")


if __name__ == '__main__':
    process()
