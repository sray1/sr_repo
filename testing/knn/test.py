# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

def unpickle(file):
    import cPickle
    fo = open(file, 'rb')
    dict = cPickle.load(fo)
    fo.close()
    return dict

unpickle('test_batch')