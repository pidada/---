#!/user/bin/env python
#- *- coding:utf-8 -*-
# user:peter


import numpy as np

def train_test_split(X, y, test_ratio=0.2, seed=None):
    """
    将数据X和y按照test_ratio分成X_train, X_test, y_train, y_test
    """
    assert X.shape[0] == y.shape[0], \
    "the size of X must be equal to the size of y"
    assert 0.0 <= test_ratio <=1.0, \
    "test_ration must be valid"
    
    # 设置随机数
    if seed:
        np.random.seed(seed)
    
    # 将X数组的长度随机排列
    shuffled_indexes = np.random.permutation(len(X))
    test_size = int(len(X) * test_ratio)
    test_indexes = shuffled_indexes[:test_size]
    train_indexes = shuffled_indexes[test_size:]
    
    X_train = X[train_indexes]
    y_train = y[train_indexes]
    
    X_test = X[test_indexes]
    y_test = y[test_indexes]
    
    return X_train, X_test, y_train, y_test