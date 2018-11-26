#!/usr/bin/python
import numpy as np
import tensorflow as tf
import matplotlib.pylab as plt
from tensorflow.examples.tutorials.mnist import input_data
import pandas as pd
import os
import glob
from Unit import bit_coding
from lstm_sequence import SpindleData
path = "datasets"
spindle = SpindleData()
datas = spindle.data
# print(spindle.data.__len__())
data_mean = []
for d in datas:
    # d = d['Time_of_night']
    data_mean.append((d[len(d)-1]-d[0])/len(d))

print(data_mean)
print(np.mean(data_mean)*3600)

# print(data)
# code =bit_coding(data, step=0.0001)
# # code = np.asarray(code)
# print(code)




