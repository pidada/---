#!/user/bin/env python
#- *- coding:utf-8 -*-
# user:peter

import numpy as np

class StandScaler:
    # 初始化过程；注意在这个类中方差用scale_表示
    def __init__(self):
        self.mean_ = None
        self.scale_ = None
    
    # 根据传进来的X来获得均值和方差；处理2维数据
    def fit(self, X):
        assert X.ndim == 2, "The dimension of X must be 2"
        # i：表示第i个属性，即第i列；shape[1]：列属性的个数；有多少个特征属性循环多少次
        self.mean_ = np.array([np.mean(X)[:, i] for i in range(X.shape[1])])
        self.scale_ = np.array([np.std(X)[:, i] for i in range(X.shape[1])])
        
        return self
    
    # 方差和均值归一化过程
    def transform(self, X):
        assert X.ndim ==2 , "the dimension of X must be 2"
        assert self.mean_ is not None and self.scale_ is not None, \
            "must fit before transform"
        assert X.shape[1] == len(self.mean_), \
            "the feature number of X must be equal to mean_ and std_"
        
        resX = np.empty(shape=X.shape, dtype=float)
        for col in range(X.shape[1]):
            resX[:, col] = (X[:, col] - self.mean_[col]) / self.scale_[col]
        return resX