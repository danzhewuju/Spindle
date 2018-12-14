#!/usr/bin/python
import Levenshtein
import numpy as np
# from lstm_sequence import SpindleData
# path = "datasets"
# spindle = SpindleData()
# datas = spindle.data
# # print(spindle.data.__len__())
# data_mean = []
# for d in datas:
#     # d = d['Time_of_night']
#     data_mean.append((d[len(d)-1]-d[0])/len(d))
#
# print(data_mean)
# print(np.mean(data_mean)*3600)

# print(data)
# code =bit_coding(data, step=0.0001)
# # code = np.asarray(code)
# print(code)
# f = open("data/cases_encoding_str.txt", 'r', encoding="UTF-8")
# data_cases = []
# for line in f:
#     data_cases.append(line.split(":")[-1])
# f.close()
# data_controls = []
# f = open("data/controls_encoding_str.txt", 'r', encoding="UTF-8")
# for line in f:
#     data_controls.append(line.split(":")[-1])
# f.close()
# jaro_cases = []
# jaro_controls = []
# for d in data_cases:
#     for d_t in data_cases:
#         jaro_cases.append(Levenshtein.jaro(d, d_t))
# for d in data_controls:
#     for d_t in data_controls:
#         jaro_controls.append(Levenshtein.jaro(d, d_t))


class A:
    a = []
    b = 0

    def __init__(self):
        # self.a.clear()
        b = np.random.randint(0, 10)
        self.a.append(b)

    def set_b(self, b):
        self.b = b

    def __del__(self):
        print("del")

        # self.a.clear()


test1 = A()
test1.set_b(11)
print(test1.a)
print(test1.b)
test1 = A()
print(test1.b)
print(test1.a)

# for i in range(10):
#     a =A()
#     print(a.a)


