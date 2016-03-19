#!/usr/bin/env python
# -- coding:utf-8 --

import os
from segmentation import Segmentation


def write_to_file(folder_name, filename, data, label):
    path = "..//"+folder_name+"_tokens/"+label
    if not os.path.exists(path):
        os.makedirs(path)
    file = open(path+"/"+filename, 'w')
    for token in data:
        token = token.encode('utf-8')
        file.write(token)
        file.write('\n')
    file.close()


def process_data(folder_name):
    data_dir = os.listdir("..//"+folder_name)
    for label in data_dir:
        for sample in os.listdir("..//"+folder_name+"/"+label):
            file = open("..//"+folder_name+"/"+label+"/"+sample, 'r')
            content = file.read()
            file.close()
            segmentation = Segmentation()
            token_list = segmentation.cut(content)
            write_to_file(folder_name, sample, token_list, label)


def process():
    process_data("train_data")
    process_data("test_data")

if __name__ == '__main__':
    process()
