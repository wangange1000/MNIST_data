# -*- coding: utf-8 -*-
"""
Created on Tue Apr 10 15:21:30 2018

@author: wangange
"""
"""
@coding: 1051739153@qq.com
"""

from keras.models import Sequential

from keras.layers import Dense, Activation, Convolution2D, MaxPooling2D, Dropout, Flatten
import keras.backend as K  # 用于重置矩阵维度
from keras.utils import np_utils

# 输入数据
from keras.datasets import mnist
filename = "D:\\MNIST_data\\test.h5"

# 一些超参数
batch_size = 128
nb_classes = 10
nb_epoch = 8

img_rows, img_cols = 28, 28
nb_filters = 32
pool_size = (2,2)
kernel_size = (3,3)

(X_train, y_train), (X_test, y_test) = mnist.load_data()

if K.image_dim_ordering() == 'th':
    # 使用Theano的顺序(conv_dim1, channels, conv_dim2, conv_dim3)
    X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
    X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    # 使用tensorflow的顺序
    X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
    X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)
    
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

# 转化向量
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

# 构建模型
model = Sequential()
model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1], border_mode='valid', input_shape=input_shape))
model.add(Activation('relu'))
model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=pool_size))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))

# 编译模型
model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])
# 拟合模型
model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch, verbose=1, validation_data=(X_test, Y_test))

# 评估模型
score = model.evaluate(X_test, Y_test, verbose=0)
print('Test score: ', score[0])
print('Test accuracy: ', score[1])

# 保存模型
model.save(filename)  # 创建HDFS 5文件