#!/user/bin/env python
#- *- coding:utf-8 -*-
# user:peter


import numpy as np

class SimpleLinearRegression1(object):
    def __init__(self):
        # a\b相当于是私有的属性
        self.a_ = None
        self.b_ = None
    
    def fit(self, x_train,y_train):
        # fit函数：根据训练数据集来得到模型
        assert x_train.ndim == 1, \
            "simple linear regression can only solve single feature training data"
        assert len(x_train) == len(y_train), \
            "the size of x_train must be equal to the size of y_train"

        x_mean = np.mean(x_train)
        y_mean = np.mean(y_train)

        num = 0.0
        d = 0.0
        for x, y in zip(x_train, y_train):
            num += (x - x_mean) * (y - y_mean)
            d += (x - x_mean) ** 2
        
        self.a_ = num / d
        self.b_ = y_mean - self.a_ * x_mean
        
        # 返回自身，sklearn对fit函数的规范
        return self
    
    def predict(self, x_predict):
        # 传进来的是待预测的x 
        assert x_predict.ndim == 1, \
            "simple linear regression can only solve single feature training data"
        assert self.a_ is not None and self.b_ is not None, \
            "must fit before predict!"
            
        return np.array([self._predict(x) for x in x_predict])
    
    def _predict(self, x_single):
        # 对一个数据进行预测 
        return self.a_ * x_single + self.b_
    
    def __repr__(self):
        # 字符串输出
        return "SimpleLinearRegression1()"
    
  
    
class SimpleLinearRegression2(object):
    def __init__(self):
        # a, b不是用户送进来的参数，相当于是私有的属性
        self.a_ = None
        self.b_ = None
    
    def fit(self, x_train, y_train):
        # fit函数：根据训练数据集来得到模型
        assert x_train.ndim == 1, \
            "simple linear regression can only solve single feature training data"
        assert len(x_train) == len(y_train), \
            "the size of x_train must be equal to the size of y_train"

        x_mean = np.mean(x_train)
        y_mean = np.mean(y_train)
        
        # 改成向量形式代替for循环，numpy中的.dot形式
        num = (x_train - x_mean).dot(y_train - y_mean)
        d = (x_train - x_mean).dot(x_train - x_mean)
        
        self.a_ = num / d
        self.b_ = y_mean - self.a_ * x_mean
        
        # 返回自身，sklearn对fit函数的规范
        return self
    
    def predict(self, x_predict):
        # 传进来的是待预测的x 
        assert x_predict.ndim == 1, \
            "simple linear regression can only solve single feature training data"
        assert self.a_ is not None and self.b_ is not None, \
            "must fit before predict!"
            
        return np.array([self._predict(x) for x in x_predict])
    
    def _predict(self, x_single):
        # 对一个数据进行预测 
        return self.a_ * x_single + self.b_
    
    def __repr__(self):
        # 字符串函数，输出方便进行查看
        return "SimpleLinearRegression2()"