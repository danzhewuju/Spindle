import matplotlib.pyplot as plt
import os
import glob
import pandas as pd
import numpy as np
import keras.preprocessing as preprocessing
import Levenshtein
import time
import threading


filter_length = 10  #设置过滤条件，数据小与这个值将会被过滤，主要是纺锤波的个数小于这个值就会这个病例就会被淘汰


class SpindleData:
    path = ""
    paths = []
    labels = []
    data = []
    cases_n = 0
    controls_n = 0
    length = 0 #固定长度的设置
    step = 0.0001#设置默认的编码间隔
    max_length = 0 #序列的最大长度
    mean_length = 0 #序列的平均长度
    coding_w = []  #元素的数据编码
    coding_q = []  #将序列弄成相同的维度

    def __init__(self, path="datasets", step=0.0001 ):
        self.path = path
        self.step =step
        self.clear_info()  #将之前旧的数据处理掉
        self.paths, self.labels = self.get_data_labels()   #获得路径以及标签
        self.coding()

    def clear_info(self):
        self.paths.clear()
        self.labels.clear()
        self.data.clear()
        self.coding_w.clear()
        self.coding_q.clear()

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
        labels = np.asarray(labels)                  #将标签转化为np的格式
        return paths, labels   #获取的是全部的文件路径

    def coding(self):#所有的数据读取以及存储(这里保存了数据的原始数据占用内存可能比较大)
        coding_q = []
        del_list = []
        sub_cases = 0   #统计病人删选的个数
        sun_control = 0  #统计正常人删选的个数
        for i, p in enumerate(self.paths):
            data = pd.read_csv(p, skiprows=(0, 1), sep=",")
            if data.__len__() < filter_length:                    #过滤掉不满足的部分
                del_list.append(i)  #记录将要删除的标签位置
                print("过滤掉了第%d个文件!" % (i+1))
                if self.labels[i] == 0:
                    sub_cases += 1
                else:
                    sun_control += 1
                continue
            print("正在读取第%d个csv文件..." % (self.paths.index(p)+1))
            data =data['Time_of_night']
            self.data.append(data)

        self.cases_n -= sub_cases        #减去被删选的数
        self.controls_n -= sun_control   #增加被删选的数
        print("cases_number:%d, controls_number:%d" % (self.cases_n, self.controls_n))
        self.labels = [x for x in self.labels if x not in del_list]  #去除掉对应的标签
        for i, d in enumerate(self.data):
            code = bit_coding(d, step=self.step)
            print("正在对第%d个序列进行编码..." %(i+1))
            coding_q.append(code)#将二位的编码加入到序列中
        self.max_length = max([len(x) for x in coding_q])
        self.mean_length = np.mean(np.asarray([len(x) for x in coding_q]))
        self.coding_w = coding_q
        # codeing_q = preprocessing.sequence.pad_sequences(codeing_q, maxlen=self.max_length)   #将所有的串都弄成相同的维度
        code_q = preprocessing.sequence.pad_sequences(coding_q, maxlen=int(self.mean_length))  # 将所有的串都弄成相同的维度
        self.coding_q = np.asarray(code_q)

    def writer_coding(self):                                         #将数据的原始编码写入到文件中（没有对齐的数据）
        f = open("./data/cases_encoding.txt", 'w', encoding="UTF-8")
        fp = open("./data/controls_encoding.txt", 'w', encoding="UTF-8")
        for index, p in enumerate(self.coding_w):
            name = self.paths[index].split('\\')[-1]
            if index < self.cases_n:
                f.write(name+" ")
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
    def trans_list_str(self, list_a):#将数组转化为字符串
        str_a = ""
        for a in list_a:
            str_a += str(a)
        return str_a

    def writing_coding_str(self):  #将对齐编码转化为字符串的形式，并写入到文件中
        f = open("./data/cases_encoding_str.txt", 'w', encoding="UTF-8")
        fp = open("./data/controls_encoding_str.txt", 'w', encoding="UTF-8")
        for index, p in enumerate(self.coding_q):
            name = self.paths[index].split('\\')[-1]
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


def bit_coding(data, step): #对一个数据进行二进制编码的实现方法
    code = []
    pre_data = 0
    count = 0
    length = len(data)
    while count < length:
        n = (data[count]-pre_data) / step
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


def get_all_paths(path):
    cate = [(os.path.join(path, x)) for x in os.listdir(path)]
    paths = []
    for i, p in enumerate(cate):
        path_tmps = glob.glob(os.path.join(p, "*.csv"))
        for p in path_tmps:
            paths.append(p)
    return paths


def get_all_data(paths):
    data = []
    for p in paths:
        d = pd.read_csv(p, seq=",", skiprows=(0, 1))
        data.append(d)
        print("Reading %d file" % (paths.index(p)+1))
    return data


def test(): #这里是测试方法
    return True

# if __name__ == '__main__':
    # test()

