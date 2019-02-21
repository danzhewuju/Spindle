#!/usr/bin/python3
import os
# 工具类


def read_all_csvfile(path):#读取一个文件夹下面所有的csv文件，不进行分类处理
    data = []
    cate = [x for x in os.listdir(path)]
