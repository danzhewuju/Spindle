#!/usr/bin/python
import numpy as np
import tensorflow as tf
import matplotlib.pylab as plt
from tensorflow.examples.tutorials.mnist import input_data
import pandas as pd
import os
import glob
import LogisticRegression
def get_Data(path, key):

    data = []
    for im in glob.glob(path + "/*.csv"):
        data_f = pd.read_csv(im, sep=",", skiprows=(0, 1))
        data_f = data_f[key]
        data.extend(data_f)
    data = np.array(data)
    return data


path_d = "datasets/cases/"
key = "Frequency"
path_n = "datasets/controls/"
data = get_Data(path_d,key)
data_n = get_Data(path_n,key)
print(data.shape)
plt.hist(data, bins=20, density=True, alpha=0.5, color="g")
plt.hist(data_n, bins=20, density=True, color="r", alpha=0.3)
plt.xlabel("Frequency")
plt.ylabel("proportion")
plt.show()
