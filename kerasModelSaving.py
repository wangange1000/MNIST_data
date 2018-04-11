# -*- coding: utf-8 -*-
"""
Created on Tue Apr 10 17:03:33 2018

@author: wangange
"""
"""
@coding: 1051739153@qq.com
"""

from keras import optimizers
from keras import objectives
from keras import metrics
from keras.models import load_model
from keras.layers import Dense, RepeatVector, TimeDistributed
from keras.models import Sequential
import numpy as np
filename = "D:\\MNIST_data\\test.h5"

def test_sequential_model_saving():
    model = Sequential()
    model.add(Dense(2, input_dim=3))
    model.add(RepeatVector(3))
    model.add(TimeDistributed(Dense(3)))
    model.compile(loss=objectives.MSE, optimizer=optimizers.RMSprop(lr=0.0001), metrics=[metrics.categorical_accuracy],sample_weight_mode='temporal')
    
    x = np.random.random((1,3))
    y = np.random.random((1,3,3))
    model.train_on_batch(x, y)
    
    out1 = model.predict(x)
    
    model.save(filename)
    
    new_model = load_model(filename)
    
    out2 = new_model.predict(x)
    print(out1)
    print(out2)
    np.testing.assert_allclose(out1, out2, atol=1e-05)  # 仅用于测试
    
if __name__=='__main__':
    print("starting...")
    test_sequential_model_saving()