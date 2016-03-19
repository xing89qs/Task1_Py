#!/usr/bin/env python
# -- coding:utf-8 --

import os


class DataSet(object):

    def __init__(self, folder_name):
        self.data_list = []
        self.label = []
        self.__read_data(folder_name)

    def __read_data(self, folder_name):
        data_dir = os.listdir("..//"+folder_name)
        for label in data_dir:
            for sample in os.listdir("..//"+folder_name+"/"+label):
                file = open("..//"+folder_name+"/"+label+"/"+sample, 'r')
                content = file.read()
                self.data_list.append(content.replace('\n', ' '))
                self.label.append(label)
                file.close()

    def __add_data(self, data):
        self.data_list.append(data)

