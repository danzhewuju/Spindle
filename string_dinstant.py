import time
from Unit import SpindleData
from Unit import calculate_distance


def test(): #这里是测试方法
    m = 1
    n = 5
    r = 0.01
    starttime = time.time()
    for i in range(m):
        print("this is %d testing"%(i+1))
        t = r * (i+1)
        path = "datasets"
        spindle = SpindleData(step=t, path=path)
        print("mean:%f" % spindle.mean_length)
        spindle.writing_coding_str()
        for j in range(n):
            print("this is %d running"%(j))
            calculate_distance()
    endtime = time.time()
    print("Running Time:%.2fs" % (endtime-starttime))
    return True


if __name__ == '__main__':
    test()
