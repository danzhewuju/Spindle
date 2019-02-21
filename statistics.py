import numpy as np
import tensorflow as tf
import matplotlib.pylab as plt
from tensorflow.examples.tutorials.mnist import input_data
import pandas as pd
import os
import glob
import glob
import pandas as pd
from LogisticRegression import get_Data

from matplotlib import pyplot as plt
run_path = "data/mros"
dataset_path = "datasets/mros_dataset"

path_mesa = "datasets/mesa_dataset"
path_mros = "datasets/mros_dataset"

#用于数据的统计


def statistic(dataset_pathmesa, dataset_pathmros):
    paths = []
    cases_mesa = []
    control_mesa = []
    data_mesa = []   # cases, control
    data_mesa.append(cases_mesa)
    data_mesa.append(control_mesa)
    cases_mros = []
    control_mros = []
    data_mros = []  # cases, control
    data_mros.append(cases_mros)
    data_mros.append(control_mros)
    cate_mesa = [(os.path.join(dataset_pathmesa, x)) for x in os.listdir(dataset_pathmesa)]
    for index, p in enumerate(cate_mesa):
        path_tmp = glob.glob(os.path.join(p, "*csv"))
        for t in path_tmp:
            d = pd.read_csv(t, sep=",")
            if index == 0:
                data_mesa[0].append(d)
            else:
                data_mesa[1].append(d)

    print("reading successfully!!!")

    cate_mros = [(os.path.join(dataset_pathmros, x)) for x in os.listdir(dataset_pathmros)]
    for index,p in enumerate(cate_mros):
        path_tmp = glob.glob(os.path.join(p, "*csv"))
        for t in path_tmp:
            d = pd.read_csv(t, sep=',', skiprows=(0, 1))
            if index == 0:
                data_mros[0].append(d)
            else:
                data_mros[1].append(d)
    return data_mesa, data_mros


data_mesa, data_mros = statistic(path_mesa, path_mros)

avg_count_mesa = 0
avg_count_mros = 0
sum_mesa = 0

data_mesa = data_mesa[1]
data_mros = data_mros[1]

count_person_mesa = data_mesa.__len__()
count_person_mros = data_mros.__len__()

# count_person_mesa = data_mesa[0].__len__()+data_mesa[1].__len__()
# count_person_mros = data_mros[0].__len__()+data_mros[1].__len__()

sum_time_mesa = 0
for d in data_mesa:
    sum_mesa += d.__len__()
    for i in range(d.__len__()):
        sum_time_mesa += d["STOP"][i] - d["Time_of_night"][i]

avg_count_mesa += sum_mesa / count_person_mesa
avg_duration_time_mesa = sum_time_mesa*3600/sum_mesa

avg_duration_time_mros = 0
sum_time_mros = 0
sum_mros = 0

for d in data_mros:
    sum_mros += d.__len__()
    for i in range(d.__len__()):
        sum_time_mros += d["Duration"][i]

avg_count_mros += sum_mros / count_person_mros
avg_duration_time_mros = sum_time_mros / sum_mros

print("avg_count_mesa=%lf, avg_count_mros=%lf" % (avg_count_mesa, avg_count_mros))
print("mesa duration time=%lf, mros duration time=%lf" % (avg_duration_time_mesa, avg_duration_time_mros))








