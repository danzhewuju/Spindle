#!/usr/python3
'''
主要是为了相关的 Accuracy，Precision，Recall
'''


class CA:
    @classmethod
    def caculate_apr(cls,tp,fp,fn,tn):
        accuracy = (tp+tn)/(tp+fp+fn+tn)
        if tp + fp == 0:
            precision = 0
        else:
            precision = tp / (tp + fp)
        recall = tp/(tp+fn)
        acc_n = tn/(tn + fp)  #正常人的准确率
        acc_p = tp/(tp + fn)  #病人的准确率
        return acc_n, acc_p, accuracy, precision, recall

