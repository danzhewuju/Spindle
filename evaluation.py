#主要是对全局的信息进行评价统计
import pandas as pd
import collections


def read_info(path):
    data = pd.read_csv(path, sep=',')
    return data


def calculation_mean(data, path, key):   #对结果进行了数据的统计
    result = {}
    dims = data["dims"]
    total_acc = data["total"]
    index = 0
    dims_count_dict = collections.Counter(dims)
    dims_count_dict = sorted(dims_count_dict.items(), key=lambda x:-x[0])
    for info in dims_count_dict:
        sum_r = 0
        for i in range(info[1]):
            sum_r += total_acc[index]
            index += 1

        result[info[0]] = sum_r / info[1]
    result = sorted(result.items(), key=lambda x: -x[1])
    f = open(path, 'a', encoding="UTF-8")
    index = key                                 #统计是第几次进行的实验结果
    for data in result:
        str_tmp = str(index)+","+str(data[0])+","+"%.5f" % (data[1])
        f.write(str_tmp+"\n")
    f.close()
    return result


if __name__ == '__main__':
    path = "data/result.csv"
    print(calculation_mean(read_info(path), "data/statistic.csv", 12))   #统计最优的字符串长度

