#!/user/bin/env python
#- *- coding:utf-8 -*-
# user:peter


import numpy as np
import matplotlib.pyplot as plt
from math import sqrt
from collections import Counter

def KNN_classify(k, X_train, y_train, x):
    
    # 确保k的范围在1到样本个数之间；
    assert 1 <= k <= X_train.shape[0], "k must be valid"
    # 样本数量和输出值的个数必须相同
    assert X_train.shape[0] == y_train.shape[0], \
    "the size of X_train and y_train must be same."
    # 确保需要预测数据的第一个维度就是属性数量
    assert X_train.shape[1] == x.shape[0], \
    "the feature number of x must be equal to X_train"
    
    # 计算欧式距离公式
    distances = [sqrt(np.sum((x_train - x)**2)) for x_train in X_train]
    nearest = np.argsort(distances)
    
    topK_y = [y_train[i] for i in nearest[:k]]
    votes =  Counter(topK_y)
    
    return votes.most_common(1)[0][0]