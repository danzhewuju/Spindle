import time
from Unit import SpindleData
import Levenshtein
import numpy as np
ratio = 0.5


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
    jaro_cases = []
    jaro_controls = []

    count = 0

    #  相互之间的比较
    # for d in data_cases:
    #     for d_t in data_cases:
    #         jaro_cases.append(Levenshtein.jaro(d, d_t))
    #         count += 1
    #         print("正在处理第{}条数据".format(count))
    # count = 0
    # for d in data_controls:
    #     for d_t in data_controls:
    #         jaro_controls.append(Levenshtein.jaro(d, d_t))
    #         count += 1
    #         print("正在处理第{}条数据".format(count))
    # ----------------------原始对齐数据------------------------


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
    m = 100
    n = 10
    r = 0.001
    starttime = time.time()
    for i in range(m):
        print("this is %d testing"%(i+1))
        t = r * (i+1)
        path = "datasets"
        spindle = SpindleData(step=t, path=path)
        # print("length:%f" % spindle.mean_length)
        print("length:%f" % spindle.max_length)
        spindle.writing_coding_str()
        for j in range(n):
            print("this is %d running"%(j))
            calculate_distance()
    endtime = time.time()
    print("Running Time:%.2fs" % (endtime-starttime))
    return True


if __name__ == '__main__':
    test()
