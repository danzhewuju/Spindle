#!/usr/python3
'''
主要是为了相关的 Accuracy，Precision，Recall
'''


class CA:
    @classmethod
    def caculate_apr(cls,tp,fp,fn,tn):
        accuracy = (tp+tn)/(tp+fp+fn+tn)
        precision = tp/(tp+fp)
        recall = tp/(tp+fn)
        return accuracy, precision, recall

