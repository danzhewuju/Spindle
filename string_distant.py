import time
from Unit import SpindleData
import Levenshtein
import numpy as np
import pandas as pd
import keras.preprocessing as preprocessing

ratio = 0.5  # 用于测试的比例


# 用于统计相关信息


def calculate_distance():  # 计算距离的评价标准是和样本字符串进行比较(非压缩啊的版本)
    # -------------------------------------------基于全部数据的存储比较-------------------------------------------
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
    data_cases_top = data_cases
    data_controls_top = data_controls  # 只是为了保持形式的一致性

    # -------------------------------------------优化top选择比较------------------------------------------
    # data_cases_top, data_controls_top = get_top_data()
    # f = open("data/cases_encoding_str.txt", 'r', encoding="UTF-8")
    # data_cases = []
    # for line in f:
    #     data_cases.append(line.split(":")[-1])
    # print("cases_encoding_str文件读取完成！")
    # f.close()
    # data_controls = []
    # f = open("data/controls_encoding_str.txt", 'r', encoding="UTF-8")
    # for line in f:
    #     data_controls.append(line.split(":")[-1])
    # print("controls_encoding_str文件读取完成！")
    # f.close()

    # --------------------------------------------选择基本的数据---------------------------------------------
    ratio_cases = np.random.randint(0, data_cases.__len__(), int(ratio * data_cases.__len__()))  # 选取20%进行测试
    ratio_control = np.random.randint(0, data_controls.__len__(), int(ratio * data_controls.__len__()))
    print("ratio_cases(count):{}, ratio_controls(count){}".format(ratio_cases.__len__(), ratio_control.__len__()))
    m = ratio_cases.__len__();
    n = ratio_control.__len__()

    Detection_queue = [data_cases[x] for x in ratio_cases] + [data_controls[x] for x in ratio_control]
    result_cases_distant = []
    result_controls_distant = []
    count = 0
    for d in Detection_queue:  # 记录病人的信息
        sum = 0
        count += 1
        for sample in data_cases_top:
            sum += Levenshtein.jaro(d, sample)
        result_cases_distant.append(sum / data_cases_top.__len__())
        print("正在处理第{}条数据...".format(count))
    count = 0
    for d in Detection_queue:
        sum = 0
        count += 1
        for sample in data_controls_top:
            sum += Levenshtein.jaro(d, sample)
        result_controls_distant.append(sum / data_controls_top.__len__())
        print("正在处理第{}条数据...".format(count))

    result_cases_distant = np.asarray(result_cases_distant)
    result_controls_distant = np.asarray(result_controls_distant)
    dim = len(data_controls[0])
    count_case = 0
    count_control = 0
    for index in range(result_controls_distant.__len__()):
        if index < m:
            if result_cases_distant[index] > result_controls_distant[index]:
                count_case += 1
            print("cases:", result_cases_distant[index], result_controls_distant[index])
        else:
            print("control:", result_cases_distant[index], result_controls_distant[index])
            if result_controls_distant[index] > result_cases_distant[index]:
                count_control += 1
    f = open("data/result.csv", 'a', encoding="UTF-8")
    result = "%d,%.4f,%.4f,%.4f\n" % (dim, count_case / m, count_control / n, (count_case + count_control) / (m + n))
    print(result)
    f.write(result)
    f.close()


def calculate_distance_compression():  # 计算距离的评价标准是和样本字符串进行比较(非压缩啊的版本)
    # -------------------------------------------优化top选择比较------------------------------------------
    data_cases_top_tmp, data_controls_top_tmp = get_top_data()
    data_cases_top = [new_str_compression(x) for x in data_cases_top_tmp]
    data_controls_top = [new_str_compression(x) for x in data_controls_top_tmp]  # 进行数据的压缩
    f = open("data/cases_encoding_str.txt", 'r', encoding="UTF-8")
    data_cases = []
    for line in f:
        data_cases.append(new_str_compression(line.split(":")[-1]))
    print("cases_encoding_str文件读取完成！")
    f.close()
    data_controls = []
    f = open("data/controls_encoding_str.txt", 'r', encoding="UTF-8")
    for line in f:
        data_controls.append(new_str_compression(line.split(":")[-1]))
    print("controls_encoding_str文件读取完成！")
    f.close()

    # --------------------------------------------选择基本的数据---------------------------------------------
    ratio_cases = np.random.randint(0, data_cases.__len__(), int(ratio * data_cases.__len__()))  # 选取20%进行测试
    ratio_control = np.random.randint(0, data_controls.__len__(), int(ratio * data_controls.__len__()))
    print("ratio_cases(count):{}, ratio_controls(count){}".format(ratio_cases.__len__(), ratio_control.__len__()))
    m = ratio_cases.__len__()
    n = ratio_control.__len__()

    Detection_queue = [data_cases[x] for x in ratio_cases] + [data_controls[x] for x in ratio_control]
    result_cases_distant = []
    result_controls_distant = []
    count = 0
    for d in Detection_queue:  # 记录病人的信息
        sum = 0
        count += 1
        for sample in data_cases_top:
            sum += Levenshtein.jaro(d, sample)
        result_cases_distant.append(sum / data_cases_top.__len__())
        print("正在处理第{}条数据...".format(count))
    count = 0
    for d in Detection_queue:
        sum = 0
        count += 1
        for sample in data_controls_top:
            sum += Levenshtein.jaro(d, sample)
        result_controls_distant.append(sum / data_controls_top.__len__())
        print("正在处理第{}条数据...".format(count))

    result_cases_distant = np.asarray(result_cases_distant)
    result_controls_distant = np.asarray(result_controls_distant)
    dim_list = [len(x) for x in data_controls] + [len(x) for x in data_cases]
    dim = np.mean(np.asarray(dim_list))
    count_case = 0
    count_control = 0
    for index in range(result_controls_distant.__len__()):
        if index < m:
            if result_cases_distant[index] > result_controls_distant[index]:
                count_case += 1
            print("cases:", result_cases_distant[index], result_controls_distant[index])
        else:
            print("control:", result_cases_distant[index], result_controls_distant[index])
            if result_controls_distant[index] > result_cases_distant[index]:
                count_control += 1
    f = open("data/result.csv", 'a', encoding="UTF-8")
    result = "%d,%.4f,%.4f,%.4f\n" % (dim, count_case / m, count_control / n, (count_case + count_control) / (m + n))
    print(result)
    f.write(result)
    f.close()
    return True


def test(flag="total"):  # 这里是测试方法
    m = 1
    n = 10
    r = 0.002  # 程序的最优化的选择
    starttime = time.time()
    for i in range(m):
        print("this is %d testing" % (i + 1))
        t = r * (i + 1)
        path = "datasets"
        spindle = SpindleData(step=t, path=path)
        spindle.set_bit_coding()
        # print("length:%f" % spindle.mean_length)   #显示的是用平均值长度还是使用最大长度
        print("length:%f" % spindle.max_length)
        spindle.writing_coding_str()
        for j in range(n):
            print("this is %d running" % (j))
            if flag == "compression":  # 如果默认的情况下是直接采用完整的字符串
                calculate_distance_compression()  # 否则采用压缩的字符串，解决稀疏性
            else:
                calculate_distance()
    endtime = time.time()
    print("Running Time:%.2fs" % (endtime - starttime))
    return True


# 获取整个样本最好的几个样本,我们认为最大的近似是最优价值的
def top_sample(ratio=0.2):
    data_cases = []
    names_cases = []
    data_controls = []
    names_controls = []
    path_cases = "data/cases_encoding_str.txt"
    f = open(path_cases, 'r', encoding="UTF-8")
    for line in f:
        data_cases.append(line.split(":")[-1])
        names_cases.append(line.split(":")[0])
    f.close()
    acc_cases = []
    for d in data_cases:
        sum = 0
        for ds in data_cases:
            sum += Levenshtein.jaro(d, ds)
        result = sum / data_cases.__len__()
        acc_cases.append(result)
    result = dict(zip(names_cases, acc_cases))
    result = sorted(result.items(), key=lambda x: -x[-1])
    number = int(data_cases.__len__() * ratio)
    # low = int(number*(0.5-ratio/2))
    # high = int(number*(0.5+ratio/2))  #用来取中位数
    f = open("data/top_cases.csv", "w", encoding="UTF-8")
    first_line = "name,acc\n"
    f.write(first_line)
    for a in range(number):
        result_tmp = "%s,%.4f\n" % (result[a][0], result[a][1])
        print(result_tmp)
        f.write(result_tmp)
    f.close()
    path_cases = "data/controls_encoding_str.txt"
    f = open(path_cases, 'r', encoding="UTF-8")
    for line in f:
        data_controls.append(line.split(":")[-1])
        names_controls.append(line.split(":")[0])
    f.close()
    acc_controls = []
    for d in data_controls:
        sum = 0
        for ds in data_controls:
            sum += Levenshtein.jaro(d, ds)
        result = sum / data_controls.__len__()
        acc_controls.append(result)
    result = dict(zip(names_controls, acc_controls))
    result = sorted(result.items(), key=lambda x: -x[-1])
    number = int(data_controls.__len__() * ratio)
    # low = int(number * (0.5 - ratio / 2))
    # high = int(number * (0.5 + ratio / 2))  # 用来取中位数
    f = open("data/top_controls.csv", "w", encoding="UTF-8")
    first_line = "name,acc\n"
    f.write(first_line)
    for a in range(number):
        result_tmp = "%s,%.4f\n" % (result[a][0], result[a][1])
        print(result_tmp)
        f.write(result_tmp)
    f.close()
    return True


# 获取由top_sample计算的结果来获取其数据
def get_top_data():
    data_cases = []
    data_controls = []
    path_top_cases = "data/top_cases.csv"
    path_top_controls = "data/top_controls.csv"
    # 获取较好样本的名称
    data = pd.read_csv(path_top_cases, sep=',')
    name_top_cases = data["name"].tolist()
    data = pd.read_csv(path_top_controls, sep=',')
    name_top_controls = data["name"].tolist()
    path_str_cases = "data/cases_encoding_str.txt"
    path_str_controls = "data/controls_encoding_str.txt"
    f = open(path_str_cases, 'r', encoding="UTF-8")
    for line in f:
        line_data = line.split(":")
        name = line_data[0]
        if name in name_top_cases:
            data_cases.append(line_data[1])
    f.close()
    f = open(path_str_controls, 'r', encoding="UTF-8")
    for line in f:
        line_data = line.split(":")
        name = line_data[0]
        if name in name_top_controls:
            data_controls.append(line_data[1])
    return data_cases, data_controls


# 主要是为了解决数据的稀疏性问题，指定一个K值，在这个K的基础上进行数据零的压缩，压缩可能会导致长度的不一致
def str_compression(data, k=8):
    result = ""
    count = 0
    for d in data:
        if d == "0" and count < k:
            count += 1
        else:
            if d == "1" and count > 0:
                result += "0" * count + "1"
                count = 0
            else:
                count = 0
                result += d
    return result


# ----------------------------------------------修改K的简单稀疏编码-------------------------------------------
# 简单稀疏编码的策略:1.当一同出现了超过k个0的时候我就手动的添加k-1个0
def new_str_compression(data, k=8):
    result = ""
    count = 0
    for d in data:
        if d == "0":
            count += 1
        else:
            if d == "1":
                if count >= 5:
                    result += "0"*(k-1)+"1"
                else:
                    result += "0"*count+"1"
                count = 0
    if count >= k:
        result += "0"*(k-1)
    else:
        result += "0"*count
    return result


def run_top_acc():  # 按照特定的规则生成代表性的字符串
    # top_sample(ratio=0.2)    #这个只需要运行一次就行主要是生成top_cases.csv,top_controls.csv文件
    test()


def test_str_compression():
    path = "data/cases_encoding_str.txt"
    f = open(path, 'r', encoding="UTF-8")
    data = f.readline().split(":")[-1]
    f.close()
    print(data)
    print("data:%d" % (data.__len__()))
    data_com = new_str_compression(data)
    print(data_com)
    print("data_com:%d" % (data_com.__len__()))


if __name__ == '__main__':
    #1 test_str_compression() #用于压缩字符串的测试
    run_top_acc()  # 生成代表性字符串
    #3 test_str = "1001000000100"
    #3 print(new_str_compression(test_str, k=5))
