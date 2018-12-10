import matplotlib.pyplot as plt
import os
import glob
import pandas as pd
import numpy as np
import keras.preprocessing as preprocessing
import Levenshtein


class SpindleData:
    path = ""
    paths = []
    labels = []
    data = []
    length = 0 #固定长度的设置
    step = 0.0001#设置默认的编码间隔
    max_length = 0 #序列的最大长度
    mean_length = 0 #序列的平均长度
    coding_w = []  #元素的数据编码
    coding_q = []  #将序列弄成相同的维度

    def __init__(self, path="datasets", step=0.0001 ):
        self.path = path
        self.step =step
        self.paths, self.labels = self.get_data_labels()   #获得路径以及标签
        self.coding()                      #

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
        labels = np.asarray(labels)                  #将标签转化为np的格式
        return paths, labels

    def coding(self):#所有的数据读取以及存储(这里保存了数据的原始数据占用内存可能比较大)
        codeing_q = []
        for p in self.paths:
            data = pd.read_csv(p, skiprows=(0, 1), sep=",")
            print("正在读取第%d个csv文件..." % (self.paths.index(p)+1))
            data =data['Time_of_night']
            self.data.append(data)
        for i, d in enumerate(self.data):
            code = bit_coding(d, step=self.step)
            print("正在对第%d个序列进行编码..."%(i+1))
            codeing_q.append(code)#将二位的编码加入到序列中
        self.max_length = max([len(x) for x in codeing_q])
        self.mean_length = np.mean(np.asarray([len(x) for x in codeing_q]))
        self.coding_w = codeing_q
        # codeing_q = preprocessing.sequence.pad_sequences(codeing_q, maxlen=self.max_length)   #将所有的串都弄成相同的维度
        codeing_q = preprocessing.sequence.pad_sequences(codeing_q, maxlen=int(self.mean_length))  # 将所有的串都弄成相同的维度
        self.coding_q = np.asarray(codeing_q)

    def writer_coding(self):                                         #将数据的原始编码写入到文件中（没有对齐的数据）
        f = open("./data/cases_encoding.txt", 'w', encoding="UTF-8")
        fp = open("./data/controls_encoding.txt", 'w', encoding="UTF-8")
        cate =[x for x in os.listdir("datasets/cases")]
        n = cate.__len__()
        for index, p in enumerate(self.coding_w):
            name = self.paths[index].split('\\')[-1]
            if index < n:
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

    def writing_coding_str(self):  #将对齐编码转化为字符串的形式
        f = open("./data/cases_encoding_str.txt", 'w', encoding="UTF-8")
        fp = open("./data/controls_encoding_str.txt", 'w', encoding="UTF-8")
        cate = [x for x in os.listdir("datasets/cases")]
        n = cate.__len__()
        for index, p in enumerate(self.coding_q):
            name = self.paths[index].split('\\')[-1]
            if index < n:
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
    # path = "datasets"
    # spindle = SpindleData(step=0.001)
    # print("mean:%f" % spindle.mean_length)
    # spindle.writing_coding_str()
    calculate_distance()
    return True


def calculate_distance():
    f = open("data/cases_encoding_str.txt", 'r', encoding="UTF-8")
    data_cases = []
    for line in f:
        data_cases.append(line.split(":")[-1])
    print("cases_encoding_str文件读取完成！")
    f.close()
    data_controls = []
    f = open("data/controls_encoding_str.txt", 'r', encoding="UTF-8")
    for line in f:
        data_controls.append(line.split(":")[-1])
    print("controls_encoding_str文件读取完成！")
    f.close()
    for d in data_cases:
        print(Levenshtein.jaro(data_cases[1], d))


if __name__ == '__main__':
    test()

