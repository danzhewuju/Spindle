import matplotlib.pyplot as plt
import os
import glob
import pandas as pd
import numpy as np
import keras.preprocessing as preprocessing


class SpindleData:
    path = ""
    paths = []
    labels = []
    data = []
    step = 0.0001
    max_length = 0#设置默认的编码间隔
    coding_q = []

    def __init__(self, path="datasets", step=0.0001 ):
        self.path = path
        self.step =step
        self.paths, self.labels = self.get_data_labels()   #获得路径以及标签
        self.coding()

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
        codeing_q = preprocessing.sequence.pad_sequences(codeing_q, maxlen=self.max_length)   #将所有的串都弄成相同的维度
        self.coding_q = np.asarray(codeing_q)


def bit_coding(data, step): #对一个数据进行编码
    code = []
    pre_data = 0
    count = 0
    length = len(data)
    while count < length:
        n = (data[count]-pre_data) / step
        if n > 0:
            if n % 1 > 0:
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


def test(): #这里是测试方法
    path = "datasets"
    spindle = SpindleData()

    return True


if __name__ == '__main__':
    test()

