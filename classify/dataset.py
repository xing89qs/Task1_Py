#!/usr/bin/env python
# -- coding:utf-8 --

import os


class DataSet(object):

    def __init__(self, folders=None):
        self.data_list = []
        self.label = []
        if folders:
            for folder_name in folders:
                self.__read_data(folder_name)

    def __read_data(self, folder_name):
        data_dir = os.listdir(folder_name)
        for label in data_dir:
            for sample in os.listdir(folder_name + "/" + label):
                file = open(folder_name + "/" + label + "/" + sample, 'r')
                content = file.read()
                self.data_list.append(content.replace('\n', ' '))
                self.label.append(label)
                file.close()
