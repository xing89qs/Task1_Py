#!/usr/bin/env python
# -- coding:utf-8 --

from feature_selection import *

FEATURE_SELECTORS = {'df': DFSelector, 'mi': MISelector, 'ig': IGSelector, 'wllr': WLLRSelector, 'chi': CHISelector,
                     'bns': BNSSelector}
