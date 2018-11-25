#!/usr/bin/python
import numpy as np
from Unit import SpindleData
from Unit import bit_coding
import tensorflow as tf
import matplotlib.pylab as plt
from tensorflow.examples.tutorials.mnist import input_data
import pandas as pd
import os
import glob


path = "datasets"
spindle = SpindleData()
# spindle.read_all_csv()
# # print(spindle.data.__len__())
# count = 0
# data_mean = []
# data = spindle.data[0]
# data = data['Time_of_night']
# # print(data)
# code =bit_coding(data, step=0.001)
# code = np.asarray(code)
# print(code)
# print(code.shape)
spindle.coding()
print(spindle.coding_q.shape)




