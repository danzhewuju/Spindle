import threading
import time
import Levenshtein
import pandas as pd
path = "data/top_cases.csv"
data = pd.read_csv(path, sep=",")
data_tmp = data["name"]
d1 = data_tmp.tolist()
print(d1)
test_str = "mros-visit1-aa5688.csv"
if test_str in data_tmp:
    print("TRUE")
