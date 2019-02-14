import threading
import time
import Levenshtein
import pandas as pd
a = [11,12,13,14 ,25,26,27,28]
b = [1, 4,2]
c = [x for i, x in enumerate(a) if i not in b]
print(c)
path ="datasets/cases/cases_nsrr6686.csv"
data = pd.read_csv(path, sep=',')
print(data)
data1 = data['Time_of_night']
print(data1)