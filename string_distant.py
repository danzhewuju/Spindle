import time
from Unit import SpindleData
import Levenshtein
import numpy as np
ratio = 0.5   #用于测试的比例
#用于统计相关信息


def calculate_distance():  #计算距离的评价标准是和所有的字符串进行比较
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
    #--------------------------------------------选择基本的数据---------------------------------------------
    ratio_cases = np.random.randint(0, data_cases.__len__(), int(ratio*data_cases.__len__()))#选取20%进行测试
    ratio_control = np.random.randint(0, data_controls.__len__(), int(ratio*data_controls.__len__()))
    print("ratio_cases(count):{}, ratio_controls(count){}".format(ratio_cases.__len__(), ratio_control.__len__()))
    m = ratio_cases.__len__() ; n = ratio_control.__len__();

    Detection_queue = [data_cases[x] for x in ratio_cases]+[data_controls[x] for x in ratio_control]
    result_cases_distant = []
    result_controls_distant = []
    count = 0
    for d in Detection_queue:   #记录病人的信息
        sum = 0
        count += 1
        for sample in data_cases:
            sum += Levenshtein.jaro(d, sample)
        result_cases_distant.append(sum/data_cases.__len__())
        print("正在处理第{}条数据...".format(count))
    count = 0
    for d in Detection_queue:
        sum = 0
        count += 1
        for sample in data_controls:
            sum += Levenshtein.jaro(d, sample)
        result_controls_distant.append(sum / data_controls.__len__())
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
    result = "%d,%.4f,%.4f,%.4f\n" % (dim, count_case/m, count_control/n, (count_case+count_control)/(m+n))
    print(result)
    f.write(result)
    f.close()


def test(): #这里是测试方法
    m = 1
    n = 1
    r = 0.002
    starttime = time.time()
    for i in range(m):
        print("this is %d testing"%(i+1))
        t = r * (i+1)
        path = "datasets"
        spindle = SpindleData(step=t, path=path)
        # print("length:%f" % spindle.mean_length)   #显示的是用平均值长度还是使用最大长度
        print("length:%f" % spindle.max_length)
        spindle.writing_coding_str()
        for j in range(n):
            print("this is %d running"%(j))
            calculate_distance()
    endtime = time.time()
    print("Running Time:%.2fs" % (endtime-starttime))
    return True


def top_sample(ratio=0.2):                  #获取整个样本最好的几个样本,我们认为最大的近似是最优价值的
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
    number = int(ratio*data_cases.__len__())
    f = open("data/top_cases.csv", "w", encoding="UTF-8")
    for a in range(number):
        result_tmp = "%s,%.4f\n" % (result[a][0], result[a][1])
        print(result_tmp)
        f.write(result_tmp)
    f.close()
    path_cases = "data/controls_encoding_str.txt"
    f = open(path_cases, 'r', encoding="UTF-8")
    first_line = "name,acc\n"
    f.write(first_line)
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
    number = int(ratio * data_controls.__len__())
    f = open("data/top_controls.csv", "w", encoding="UTF-8")
    first_line = "name,acc\n"
    f.write(first_line)
    for a in range(number):
        result_tmp = "%s,%.4f\n" % (result[a][0], result[a][1])
        print(result_tmp)
        f.write(result_tmp)
    f.close()


def get_top_data():#获取由top_sample计算的结果来获取其数据
    data_cases = []
    data_controls = []
    return data_cases, data_controls


def calculate_step():  #通过字符串的长度来计算出对应的步长
    step = 0.0001
    n = 100
    f = open("data/tran_length_step.csv", 'a', encoding="UTF-8")
    for i in range(1, n+1):
        step_tem = step*i
        spindle = SpindleData(step=i)
        result = str(spindle.max_length)+","+str(step_tem) + "," + str(3600*step_tem)+"\n"
        f.write(result)  #最大长度的时间间隔映射表
        print("Writing Success!")
    f.close()
    return True


if __name__ == '__main__':
    # calculate_step()
    test()
    # top_sample()
