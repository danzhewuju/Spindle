import matplotlib.pyplot as plt
import os
import glob
import pandas as pd
import numpy as np
import keras.preprocessing as preprocessing
import math
import Levenshtein
import time
import threading

filter_length = 10  # 设置过滤条件，数据小与这个值将会被过滤，主要是纺锤波的个数小于这个值就会这个病例就会被淘汰
run_path = "data/mesa"    #程序运行的路径,实验结果的保存
dataset_path = "datasets/mesa_dataset/"  #实验中原始数据存放位置


class SpindleData:
    path = ""
    paths = []
    labels = []
    names = []  # 对应的文件名称列表
    data = []  # Time_of_night序列中的所有数据
    cases_n = 0
    controls_n = 0
    length = 0  # 固定长度的设置
    step = 0.002  # 设置默认的编码间隔
    max_length = 0  # 序列的最大长度
    mean_length = 0  # 序列的平均长度
    coding_w = []  # 元素的数据编码,字符串的形式eg.10010010001...
    coding_q = []  # 将序列弄成相同的维度，二值向量的序列[0,1,1,0,...]
    coding_number_distribution = []  # 在特定步长中纺锤波的个数分布(长度可能不一致)
    coding_number_distribution_isometic = []  # 纺锤波个数分布的对齐操作

    def __init__(self, path=dataset_path, step=0.002):
        self.clear_info()  # 将之前旧的数据处理掉
        self.path = path
        self.step = step
        self.paths, self.labels = self.get_data_labels()  # 获得路径以及标签
        self.coding_setting()

    def clear_info(self):
        self.paths.clear()
        self.labels.clear()
        self.data.clear()
        self.coding_w.clear()
        self.coding_q.clear()

    def get_spindle_number_distribution(self, ismax_length=True):
        code_list = []
        for index, d in enumerate(self.data):
            code = num_coding(d, self.step)
            code_list.append(np.asarray(code))
            print("正在统计第%d数据:%s的相关信息" % (index, self.names[index]))
        self.coding_number_distribution = code_list
        if ismax_length:
            code_length = max([len(x) for x in code_list])
            print("max_length:%d" % code_length)
        else:
            code_length = int(np.mean(np.asarray([len(x) for x in code_list])))  # 长度设置为均值
            print("mean_length:%d" % code_length)
        code_final = preprocessing.sequence.pad_sequences(code_list, maxlen=code_length)
        self.coding_number_distribution_isometic = code_final  # 个数分布的对齐操作
        return self.coding_number_distribution

    def get_data_labels(self):  # 返回获取的数据以及标签[0,1,0,1,...]  "./datasets/"
        path = self.path
        cate = [(os.path.join(path, x)) for x in os.listdir(path)]
        paths = []
        labels = []
        for i, p in enumerate(cate):
            path_tmps = glob.glob(os.path.join(p, "*.csv"))
            for p in path_tmps:
                paths.append(p)
                labels.append(i)
                if i == 0:
                    self.cases_n += 1
                else:
                    self.controls_n += 1
        labels = np.asarray(labels)  # 将标签转化为np的格式
        return paths, labels  # 获取的是全部的文件路径

    def coding_setting(self):  # 所有的数据读取以及存储(这里保存了数据的原始数据占用内存可能比较大)
        del_list = []
        sub_cases = 0  # 统计病人删选的个数
        sun_control = 0  # 统计正常人删选的个数
        for i, p in enumerate(self.paths):
            if dataset_path == "datasets/mesa_dataset/":
                data = pd.read_csv(p, sep=",")  # 第二个数据集
            else:
                data = pd.read_csv(p, skiprows=(0, 1), sep=",")#第一个数据集，格式不相同
            if data.__len__() < filter_length:  # 过滤掉不满足的部分
                del_list.append(i)  # 记录将要删除的标签位置
                print("过滤掉了第%d个文件!" % (i + 1))
                if self.labels[i] == 0:
                    sub_cases += 1
                else:
                    sun_control += 1
                continue
            print("正在读取第%d个csv文件..." % (self.paths.index(p) + 1))
            data = data['Time_of_night']
            self.data.append(data)

        self.cases_n -= sub_cases  # 减去被删选的数
        self.controls_n -= sun_control  # 增加被删选的数
        self.labels = [x for i, x in enumerate(self.labels) if i not in del_list]  # 去除掉对应的标签
        self.names = [x.split("\\")[-1] for i, x in enumerate(self.paths) if i not in del_list]  # windows 下的文件名称提取
        # self.names = [x.split("/")[-1] for i, x in enumerate(self.paths) if i not in del_list]  #mac 下的文件名字的提取
        print("cases_n:%d, controls_n:%d, total:%d"%(self.cases_n, self.controls_n, self.data.__len__()))
        return True

    def set_bit_coding(self):  # 二进制的编码(0,1,1,1,1,1,0,0,0)
        coding_q = []
        for i, d in enumerate(self.data):
            code = bit_coding(d, step=self.step)
            print("正在对第%d个序列进行编码..." % (i + 1))
            coding_q.append(code)  # 将二位的编码加入到序列中
        self.max_length = max([len(x) for x in coding_q])
        self.mean_length = np.mean(np.asarray([len(x) for x in coding_q]))
        self.coding_w = coding_q
        code_q = preprocessing.sequence.pad_sequences(coding_q, maxlen=self.max_length)  # 将所有的串都弄成相同的维度(最大长度)
        # code_q = preprocessing.sequence.pad_sequences(coding_q, maxlen=int(self.mean_length))  # 将所有的串都弄成相同的维度(平均长度)
        self.coding_q = np.asarray(code_q)

    def set_sub_type_coding(self):  # 带亚型的编码（0,1,2,1,2,2,1,2）
        sub_type_data = sub_type_spindle()  #获得文件下亚型和名字的字典映射关系
        coding_q = []
        for i, d in enumerate(self.data):
            name = "membership_"+ self.names[i]
            code = sub_type_coding(d, sub_type_data[name], step=self.step)
            print("正在对第%d个序列进行编码..." % (i + 1))
            coding_q.append(code)  # 将二位的编码加入到序列中
        self.max_length = max([len(x) for x in coding_q])
        self.mean_length = np.mean(np.asarray([len(x) for x in coding_q]))
        self.coding_w = coding_q
        code_q = preprocessing.sequence.pad_sequences(coding_q, maxlen=self.max_length)  # 将所有的串都弄成相同的维度(最大长度)
        # code_q = preprocessing.sequence.pad_sequences(coding_q, maxlen=int(self.mean_length))  # 将所有的串都弄成相同的维度(平均长度)
        self.coding_q = np.asarray(code_q)

    def writer_coding(self):  # 将数据的原始编码写入到文件中（没有对齐的数据）
        f_path = run_path + "/cases_encoding.txt"
        fp_path = run_path + "/controls_encoding.txt"
        f = open(f_path, 'w', encoding="UTF-8")
        fp = open(fp_path, 'w', encoding="UTF-8")
        for index, p in enumerate(self.coding_w):
            # name = self.paths[index].split('\\')[-1]
            name = self.paths[index].split('/')[-1] #mac 下的文件名的提取
            if index < self.cases_n:
                f.write(name + " ")
                f.writelines(str(p))
                f.write("\n")
            else:
                fp.write(name + " ")
                fp.writelines(str(p))
                fp.write("\n")
        f.close()
        fp.close()
        print("Writing Success!!!")

    @classmethod
    def trans_list_str(self, list_a):  # 将数组转化为字符串
        str_a = ""
        for a in list_a:
            str_a += str(a)
        return str_a

    def writing_coding_str(self):  # 将对齐编码转化为字符串的形式，并写入到文件中
        f_path = run_path + "/cases_encoding_str.txt"
        fp_path = run_path + "/controls_encoding_str.txt"
        f = open(f_path, 'w', encoding="UTF-8")
        fp = open(fp_path, 'w', encoding="UTF-8")
        for index, p in enumerate(self.coding_q):
            name = self.names[index]
            if index < self.cases_n:
                f.write(name + ":")
                str_a = SpindleData.trans_list_str(p)
                f.writelines(str_a)
                f.write("\n")
            else:
                fp.write(name + ":")
                str_a = SpindleData.trans_list_str(p)
                fp.writelines(str_a)
                fp.write("\n")
        f.close()
        fp.close()
        print("Writing Success!!!")


# 基于个数的二进制编码
def bit_coding(data, step):  # 对一个数据进行二进制编码的实现方法,data:一个病人的序列信息 step:所选择步长
    code = []
    pre_data = 0
    count = 0
    length = len(data)
    while count < length:
        n = (data[count] - pre_data) / step
        if n > 0:
            if n > int(n):
                n = int(n)
                code += [0] * n + [1]
            else:
                n = int(n)
                code += [0] * (n - 1) + [1]
        pre_data = data[count]
        count += 1
    return code


'''
----------------------------------------------一个独立的模块用来处理亚型---------------------------------------------
处理机制：增加的数据中每一个纺锤波都对应一个亚型，但我在记录数据的时候我会用这个亚型来替代原来的编码1
'''


def sub_type_spindle(path=run_path+"sub_spindle_type"):  # "sub_spindle_type"
    # 文件模块的读取
    cate = [os.path.join(path, x) for x in os.listdir(path)]
    cates = []
    names = []
    for p in cate:
        for pt in os.listdir(p):
            cates.append(os.path.join(p, pt))
    paths = []
    for i, p in enumerate(cates):
        path_tmps = glob.glob(os.path.join(p, "*.csv"))
        for name in os.listdir(p):
            names.append(name)
        for p in path_tmps:
            paths.append(p)
    #获得了文件的路径和名称，形成映射关系
    sub_type_data = []
    for p in paths:
        data = pd.read_csv(p, sep=',')
        sub_type_data.append(data["membership"])
    # print(sub_type_data)
    sub_type_dic = dict(zip(names, list(sub_type_data)))
    # print(sub_type_dic)
    return sub_type_dic


'''添加了纺锤波亚型的编码方式 '''
def sub_type_coding(data, sub_data, step):  #分别还是需要编码的序列，亚型样本，步长
    code = []
    pre_data = 0
    count = 0
    sub_type_index = 0
    length = len(data)
    while count < length:
        n = (data[count] - pre_data) / step
        if n > 0:
            if n > int(n):
                n = int(n)
                code += [0] * n + [sub_data[sub_type_index]]
            else:
                n = int(n)
                code += [0] * (n - 1) + [sub_data[sub_type_index]]
            sub_type_index += 1
        pre_data = data[count]
        count += 1
    return code


# 基于个数分布的编码方式
def num_coding(data, step):
    code = []
    pre_flag = step  # 表示的是前步节点
    count = 0
    write_count = 0  # 每一个区间内的个数记录
    length = len(data)
    while count < length:
        if data[count] > pre_flag:
            code.append(write_count)
            pre_flag += step  # 提升它的上界
            write_count = 0
        else:
            write_count += 1
            count += 1
    if write_count != 0:
        code.append(write_count)
    return code


# 两个长序列(相同维度)的乘法
def multiply(data1, data2):
    length = len(data1)
    sum = 0
    for index in range(length):
        sum += data1[index] * data2[index]
    return sum


# 余弦相似度的计算
def cos(data1, data2):
    d1 = multiply(data1, data2)
    d2 = math.sqrt(multiply(data1, data1)) * math.sqrt(multiply(data2, data2))
    result = d1 / d2
    return result


def draw(history):
    # 图形作图
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(1, len(acc) + 1)

    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and Validation accuracy')
    plt.legend()
    plt.figure()

    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and Validation loss')
    plt.legend()
    plt.show()


# def get_all_paths(path):
#     cate = [(os.path.join(path, x)) for x in os.listdir(path)]
#     paths = []
#     for i, p in enumerate(cate):
#         path_tmps = glob.glob(os.path.join(p, "*.csv"))
#         for p in path_tmps:
#             paths.append(p)
#     return paths


# def get_all_data(paths):
#     data = []
#     for p in paths:
#         d = pd.read_csv(p, seq=",", skiprows=(0, 1))
#         data.append(d)
#         print("Reading %d file" % (paths.index(p)+1))
#     return data


# def test(): #这里是测试方
# spindle = SpindleData(step=0.002)
# spindle.writing_coding_str()
# spindle = SpindleData(step=0.25)
# code = spindle.get_spindle_number_distribution()
# code_max_length = max([len(x) for x in code])
# code_final = preprocessing.sequence.pad_sequences(code, maxlen=code_max_length)
# print(code)
# print(code_final)
# result = cos(code_final[0], code_final[1])
# print(result)
# return True

if __name__ == '__main__':
#     # name = "membership_mros-visit1-aa0121.csv"
#     # data = sub_type_spindle("sub_spindle_type")
#     # print(list(data[name]))
#     # test = [2, 10, 15, 16, 20]
#     # sub_data = [1, 5, 4, 3, 3]
#     # print(sub_type_coding(test,sub_data, step=2))
    spindle = SpindleData(step=0.001)

    # spindle.set_sub_type_coding()
    spindle.set_bit_coding()

    # print(spindle.coding_q[0])
    spindle.writing_coding_str()
