#!/usr/bin/env python
# -- coding:utf-8 --

from classify import DataSet
import numpy as np
import pandas as pd
from collections import defaultdict
import math
from sklearn.feature_selection import chi2, SelectKBest
from scipy.stats import norm


class FeatureSelector(object):
    def __init__(self):
        pass

    def select(self, data_set):
        pass


class DFSelector(FeatureSelector):
    def __init__(self):
        pass

    def get_val(self, data_set):
        df_list = []
        for data in data_set.data_list:
            tokens = set([_ for _ in data.split()])
            df_list.extend(list(tokens))
        df_series = pd.Series(df_list)
        ret = df_series.value_counts()
        result = dict()
        for _, __ in ret.iteritems():
            result[_] = __
        return result

    # num -- choose num of tokens
    def select(self, data_set, num=5000):
        new_set = DataSet()
        new_set.label = data_set.label
        _DF_Series = self.get_val(data_set)
        _DF_Series = pd.Series(_DF_Series)
        _DF_Series.sort_values(ascending=False, inplace=True)
        vocabulary = set(_DF_Series[0:min(num, len(_DF_Series))].index)
        for data in data_set.data_list:
            tokens = [_ for _ in data.split() if _ in vocabulary]
            new_set.data_list.append(" ".join(tokens))
        return new_set


class IGSelector(FeatureSelector):
    def __init__(self):
        pass

    def get_val(self, data_set):
        sample_len = len(data_set.data_list)
        class_mp = defaultdict()
        _K = 0
        for _ in data_set.label:
            if _ not in class_mp:
                class_mp[_] = _K
                _K += 1
        p_dict = [defaultdict(int) for _ in xrange(_K)]
        class_num = np.zeros(_K)
        for _i in xrange(len(data_set.data_list)):
            data = data_set.data_list[_i]
            tokens = set([_ for _ in data.split()])
            for _ in tokens:
                p_dict[class_mp[data_set.label[_i]]][_] += 1
            class_num[class_mp[data_set.label[_i]]] += 1
        _IG_Series = dict()
        for _i in xrange(_K):
            for key, value in p_dict[_i].iteritems():
                p1 = float(value) / class_num[_i]
                if value == 0:
                    continue
                token_num = 0
                for _j in xrange(_K):
                    token_num += p_dict[_j][key]
                p2 = float(token_num) / sample_len
                # print p1, p2
                value = math.log(p1 / p2)
                value_bar = math.log((1.0 - p1) / (1.0 - p2))
                if key in _IG_Series:
                    _IG_Series[key] += p1 * value + (1.0 - p1) * value_bar
                else:
                    _IG_Series[key] = p1 * value + (1.0 - p1) * value_bar
        return _IG_Series

    def select(self, data_set, num=5000):
        new_set = DataSet()
        new_set.label = data_set.label
        _IG_Series = self.get_val(data_set)
        _IG_Series = pd.Series(_IG_Series)
        _IG_Series.sort_values(ascending=False, inplace=True)
        vocabulary = set(_IG_Series[0:min(num, len(_IG_Series))].index)
        for data in data_set.data_list:
            tokens = [_ for _ in data.split() if _ in vocabulary]
            new_set.data_list.append(" ".join(tokens))
        return new_set


class MISelector(FeatureSelector):
    def __init__(self):
        pass

    def get_val(self, data_set, type='max'):
        sample_len = len(data_set.data_list)
        class_mp = defaultdict()
        _K = 0
        for _ in data_set.label:
            if _ not in class_mp:
                class_mp[_] = _K
                _K += 1
        p_dict = [defaultdict(int) for _ in xrange(_K)]
        class_num = np.zeros(_K)
        for _i in xrange(len(data_set.data_list)):
            data = data_set.data_list[_i]
            tokens = set([_ for _ in data.split()])
            for _ in tokens:
                p_dict[class_mp[data_set.label[_i]]][_] += 1
            class_num[class_mp[data_set.label[_i]]] += 1
        _MI_Series = dict()
        for _i in xrange(_K):
            for key, value in p_dict[_i].iteritems():
                p1 = float(value) / class_num[_i]
                if value == 0:
                    continue
                token_num = 0
                for _j in xrange(_K):
                    token_num += p_dict[_j][key]
                p2 = float(class_num[_i]) / sample_len
                # print p1, p2
                value = math.log(p1 / p2)
                if key in _MI_Series:
                    if type == 'max':
                        _MI_Series[key] = max(_MI_Series[key], value)
                    elif type == 'avg':
                        _MI_Series[key] += float(class_num[_i]) / sample_len * value
                else:
                    if type == 'max':
                        _MI_Series[key] = value
                    elif type == 'avg':
                        _MI_Series[key] = float(class_num[_i]) / sample_len * value
        return _MI_Series


    def select(self, data_set, type='max', num=5000):
        new_set = DataSet()
        new_set.label = data_set.label
        _MI_Series = self.get_val(data_set, type=type)
        _MI_Series = pd.Series(_MI_Series)
        _MI_Series.sort_values(ascending=False, inplace=True)
        vocabulary = set(_MI_Series[0:min(num, len(_MI_Series))].index)
        for data in data_set.data_list:
            tokens = [_ for _ in data.split() if _ in vocabulary]
            new_set.data_list.append(" ".join(tokens))
        return new_set


class CHISelector(FeatureSelector):
    def __init__(self):
        pass

    def get_val(self, data_set):
        sample_len = len(data_set.data_list)
        class_mp = defaultdict()
        _K = 0
        for _ in data_set.label:
            if _ not in class_mp:
                class_mp[_] = _K
                _K += 1
        p_dict = [defaultdict(int) for _ in xrange(_K)]
        class_num = np.zeros(_K)
        for _i in xrange(len(data_set.data_list)):
            data = data_set.data_list[_i]
            tokens = set([_ for _ in data.split()])
            for _ in tokens:
                p_dict[class_mp[data_set.label[_i]]][_] += 1
            class_num[class_mp[data_set.label[_i]]] += 1
        _CHI_Series = dict()
        for _i in xrange(_K):
            for key, value in p_dict[_i].iteritems():
                token_num = 0
                for _j in xrange(_K):
                    token_num += p_dict[_j][key]
                A = value
                B = token_num - A
                N_all = sample_len
                N_i = class_num[_i]
                # print p1, p2
                value = float(N_all * ((A * (N_all - N_i - B) - (N_i - A) * B) ** 2)) / float(
                    N_i * (N_all - N_i) * (A + B) * (N_all - A - B))
                if key in _CHI_Series:
                    _CHI_Series[key] = max(_CHI_Series[key], value)
                else:
                    _CHI_Series[key] = value
        return _CHI_Series

    def select(self, data_set, num=5000):
        new_set = DataSet()
        new_set.label = data_set.label
        _CHI_Series = self.get_val(data_set)
        _CHI_Series = pd.Series(_CHI_Series)
        _CHI_Series.sort_values(ascending=False, inplace=True)
        vocabulary = set(_CHI_Series[0:min(num, len(_CHI_Series))].index)
        for data in data_set.data_list:
            tokens = [_ for _ in data.split() if _ in vocabulary]
            new_set.data_list.append(" ".join(tokens))
        return new_set


class BNSSelector(FeatureSelector):
    def __init__(self):
        pass

    def get_val(self, data_set):
        sample_len = len(data_set.data_list)
        class_mp = defaultdict()
        _K = 0
        for _ in data_set.label:
            if _ not in class_mp:
                class_mp[_] = _K
                _K += 1
        p_dict = [defaultdict(int) for _ in xrange(_K)]
        class_num = np.zeros(_K)
        for _i in xrange(len(data_set.data_list)):
            data = data_set.data_list[_i]
            tokens = set([_ for _ in data.split()])
            for _ in tokens:
                p_dict[class_mp[data_set.label[_i]]][_] += 1
            class_num[class_mp[data_set.label[_i]]] += 1
        _BNS_Series = dict()
        for _i in xrange(_K):
            for key, value in p_dict[_i].iteritems():
                p1 = float(value) / class_num[_i]
                if value == 0:
                    continue
                p2 = float(value) / (sample_len - class_num[_i])
                # print p1, p2
                value = math.fabs(norm.ppf(p1) - norm.ppf(p2))  # 逆正态分布函数,有点慢
                if key in _BNS_Series:
                    _BNS_Series[key] = max(_BNS_Series[key], value)
                else:
                    _BNS_Series[key] = value
        return _BNS_Series

    def select(self, data_set, num=5000):
        new_set = DataSet()
        new_set.label = data_set.label
        _BNS_Series = self.get_val(data_set)
        _BNS_Series = pd.Series(_BNS_Series)
        _BNS_Series.sort_values(ascending=False, inplace=True)
        vocabulary = set(_BNS_Series[0:min(num, len(_BNS_Series))].index)
        for data in data_set.data_list:
            tokens = [_ for _ in data.split() if _ in vocabulary]
            new_set.data_list.append(" ".join(tokens))
        return new_set


class WLLRSelector(FeatureSelector):
    def __init__(self):
        pass

    def get_val(self, data_set):
        sample_len = len(data_set.data_list)
        class_mp = defaultdict()
        _K = 0
        for _ in data_set.label:
            if _ not in class_mp:
                class_mp[_] = _K
                _K += 1
        p_dict = [defaultdict(int) for _ in xrange(_K)]
        class_num = np.zeros(_K)
        for _i in xrange(len(data_set.data_list)):
            data = data_set.data_list[_i]
            tokens = set([_ for _ in data.split()])
            for _ in tokens:
                p_dict[class_mp[data_set.label[_i]]][_] += 1
            class_num[class_mp[data_set.label[_i]]] += 1
        _WLLR_Series = dict()
        for _i in xrange(_K):
            for key, value in p_dict[_i].iteritems():
                p1 = float(value) / class_num[_i]
                if value == 0:
                    continue
                p2 = float(value) / (sample_len - class_num[_i])
                # print p1, p2
                value = math.log(p1 / p2) * p1
                if key in _WLLR_Series:
                    _WLLR_Series[key] = max(_WLLR_Series[key], value)
                else:
                    _WLLR_Series[key] = value
        return _WLLR_Series

    def select(self, data_set, num=5000):
        new_set = DataSet()
        new_set.label = data_set.label
        _WLLR_Series = self.get_val(data_set)
        _WLLR_Series = pd.Series(_WLLR_Series)
        _WLLR_Series.sort_values(ascending=False, inplace=True)
        vocabulary = set(_WLLR_Series[0:min(num, len(_WLLR_Series))].index)
        for data in data_set.data_list:
            tokens = [_ for _ in data.split() if _ in vocabulary]
            new_set.data_list.append(" ".join(tokens))
        return new_set


class AllSelector(FeatureSelector):
    def __init__(self):
        pass

    def normal(self, series):
        mx = -100000000000000.0
        mi = -100000000000000.0
        for key, val in series.iteritems():
            mx = max(mx, val)
            mi = min(mi, val)
        for key in series.keys():
            pre = series[key]
            now = (mx - pre) / (mx - mi)
            series[pre] = now
        return series

    def select(self, data_set, num=5000):
        new_set = DataSet()
        new_set.label = data_set.label
        _ALL_Series = dict()
        _SELECTORS = [DFSelector, IGSelector, MISelector, CHISelector, BNSSelector, WLLRSelector]
        for selector in _SELECTORS:
            selector = selector()
            _Series = self.normal(selector.get_val(data_set))
            for key, val in _Series.iteritems():
                if key in _ALL_Series:
                    _ALL_Series[key] = max(_ALL_Series[key], val)
                else:
                    _ALL_Series[key] = val
        _ALL_Series = pd.Series(_ALL_Series)
        _ALL_Series.sort_values(ascending=False, inplace=True)
        vocabulary = set(_ALL_Series[0:min(num, len(_ALL_Series))].index)
        for data in data_set.data_list:
            tokens = [_ for _ in data.split() if _ in vocabulary]
            new_set.data_list.append(" ".join(tokens))
        return new_set
