# !/user/bin/env python
# - *- coding:utf-8 -*-
# user:peter

import numpy as np
from math import sqrt
from collections import Counter
from metrics import accuracy_score


class KNNClassifier:
    def __init__(self, k):
        # 构造函数：初始化KNN分类器，传入k值；
        # 将样本定义为私有属性None，外部无法变动
        assert k >= 1, "k must be valid"
        self.k = k
        self._X_train = None
        self._y_train = None
        
    def fit(self, X_train, y_train):
        # 样本数量X_train和输出值的个数必须相同；每个样本实例对应一个结果y
        assert X_train.shape[0] == y_train.shape[0], \
            "the size of X_train and y_train must be same."
        # k <= 总样本数（x的shape属性第一个代表样本总数；第二个代表样本属性的个数）
        assert self.k <= X_train.shape[0], \
            "the feature number of x must be equal to X_train"

        # 传入已知训练数据(X_train, y_train)，并且返回
        self._X_train = X_train
        self._y_train = y_train
        return self
    
    def predict(self, X_predict):
        # 给定待预测数据集X_predict, 返回表示预测X_predict的结果向量
        
        # 传入样本(_X_train, _y_train)都不能是空值，fit函数中已经赋值
        assert self._X_train is not None and self._y_train is not None, \
        "must fit before predict"
        # 样本数不限制，预测数据的属性和训练X_train的属性个数必须相同
        assert X_predict.shape[1] == self._X_train.shape[1], \
        "the feature number of X_predict  must be equal to X_train"
        
        # _predict对一个向量进行预测
        y_predict = [self._predict(x) for x in X_predict]
        return np.array(y_predict)
    
    def _predict(self, x):
        # 给出需要预测的数据的特征数量等于原来的特征数量，返回x预测的结果
        # 单个待预测数据的shape属性第一个值即为训练数据X_train的特征属性个数
        assert x.shape[0] == self._X_train.shape[1], \
        "the feature number of x must be equal to X_train"
        
        distances = [sqrt(np.sum((x_train - x)**2)) for x_train in self._X_train]
        nearest = np.argsort(distances)
        
        topK_y = [self._y_train[i] for i in nearest[:self.k]]
        votes = Counter(topK_y)
        
        return votes.most_common(1)[0][0]
    
    def score(self, X_test, y_test):
        '''传入X_test预测得到y_predict，再和y_test传入准确率的函数中：'''
        y_predict = self.predict(X_test)
        return accuracy_score(y_test, y_predict)
    
    
    def __repr__(self):
        return "KNN(k={})".format(self.k)