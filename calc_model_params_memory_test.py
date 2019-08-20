# -*- coding: utf-8 -*-
"""
Created on Sun Feb 17 20:26:51 2019
计算模型的参数量和内存占用
参考：https://blog.csdn.net/u011311291/article/details/82969409
@author: liuhuaqing
"""

'''
Created on 2018年9月30日

'''

from keras import applications
import numpy as np

import cv2
#image = cv2.imread("D:\\xxxx\\hashiqi.jpg")
#image = cv2.resize(image,(1024,1024),interpolation = cv2.INTER_CUBIC)
#x_train = np.expand_dims(image,axis=0)
#y_train = np.array([0])
#print(image.shape)
# (1024, 1024, 3)

model = applications.VGG16(input_shape=(1024,1024,3),include_top=False,weights=None)
print("无全连接层总参数量:",model.count_params())
#model = applications.VGG16(input_shape=(1024,1024,3),include_top=True,weights=None)
#print("有全连接层总参数量:",model.count_params())
# 无全连接层总参数量: 14714688
# 有全连接层总参数量: 2183080744 可见权重都基本占用在全连接层

all_params_memory = 0
all_feature_memory = 0
for layer in model.layers:
    #训练权重w占用的内存
    params_memory = layer.count_params()/(1024*1024) * 4
    print("训练权重w占用的内存:",layer.name,layer.count_params(),str(params_memory)+" M")
    all_params_memory = all_params_memory + params_memory
    #特征图占用内存
    feature_shape = layer.output_shape
    feature_size = 1
    for i in range(1,len(feature_shape)):
        feature_size = feature_size*feature_shape[i]
    feature_memory = feature_size/(1024*1024) * 4
    print("特征图占用内存:",feature_shape,feature_size,str(feature_memory)+" M")
    all_feature_memory = all_feature_memory + feature_memory

# 特征图占用内存: (None, 1024, 1024, 3) 3145728 12.0 M
# 训练权重w占用的内存: block1_conv1 1792 0.0068359375 M
# 特征图占用内存: (None, 1024, 1024, 64) 67108864 256.0 M
# 训练权重w占用的内存: block1_conv2 36928 0.140869140625 M
# 特征图占用内存: (None, 1024, 1024, 64) 67108864 256.0 M
# 训练权重w占用的内存: block1_pool 0 0.0 M
# 特征图占用内存: (None, 512, 512, 64) 16777216 64.0 M
# 训练权重w占用的内存: block2_conv1 73856 0.28173828125 M
# 特征图占用内存: (None, 512, 512, 128) 33554432 128.0 M
# 训练权重w占用的内存: block2_conv2 147584 0.56298828125 M
# 特征图占用内存: (None, 512, 512, 128) 33554432 128.0 M
# 训练权重w占用的内存: block2_pool 0 0.0 M
# 特征图占用内存: (None, 256, 256, 128) 8388608 32.0 M
# 训练权重w占用的内存: block3_conv1 295168 1.1259765625 M
# 特征图占用内存: (None, 256, 256, 256) 16777216 64.0 M
# 训练权重w占用的内存: block3_conv2 590080 2.2509765625 M
# 特征图占用内存: (None, 256, 256, 256) 16777216 64.0 M
# 训练权重w占用的内存: block3_conv3 590080 2.2509765625 M
# 特征图占用内存: (None, 256, 256, 256) 16777216 64.0 M
# 训练权重w占用的内存: block3_pool 0 0.0 M
# 特征图占用内存: (None, 128, 128, 256) 4194304 16.0 M
# 训练权重w占用的内存: block4_conv1 1180160 4.501953125 M
# 特征图占用内存: (None, 128, 128, 512) 8388608 32.0 M
# 训练权重w占用的内存: block4_conv2 2359808 9.001953125 M
# 特征图占用内存: (None, 128, 128, 512) 8388608 32.0 M
# 训练权重w占用的内存: block4_conv3 2359808 9.001953125 M
# 特征图占用内存: (None, 128, 128, 512) 8388608 32.0 M
# 训练权重w占用的内存: block4_pool 0 0.0 M
# 特征图占用内存: (None, 64, 64, 512) 2097152 8.0 M
# 训练权重w占用的内存: block5_conv1 2359808 9.001953125 M
# 特征图占用内存: (None, 64, 64, 512) 2097152 8.0 M
# 训练权重w占用的内存: block5_conv2 2359808 9.001953125 M
# 特征图占用内存: (None, 64, 64, 512) 2097152 8.0 M
# 训练权重w占用的内存: block5_conv3 2359808 9.001953125 M
# 特征图占用内存: (None, 64, 64, 512) 2097152 8.0 M
# 训练权重w占用的内存: block5_pool 0 0.0 M
# 特征图占用内存: (None, 32, 32, 512) 524288 2.0 M
# 训练权重w占用的内存: flatten 0 0.0 M
# 特征图占用内存: (None, 524288) 524288 2.0 M
# 训练权重w占用的内存: fc1 2147487744 8192.015625 M
# 特征图占用内存: (None, 4096) 4096 0.015625 M
# 训练权重w占用的内存: fc2 16781312 64 .015625 M
# 特征图占用内存: (None, 4096) 4096 0.015625 M
# 训练权重w占用的内存: predictions 4097000 15.628814697265625 M
# 特征图占用内存: (None, 1000) 1000 0.003814697265625 M
    
print("网络权重W占用总内存:",str(all_params_memory)+" M")
print("网络特征图占用总内存:",str(all_feature_memory)+" M")
print("网络总消耗内存:",str(all_params_memory+all_feature_memory)+" M")
# 网络权重W占用总内存: 8327.79214477539 M
# 网络特征图占用总内存: 1216.0350647 M
# 网络总消耗内存: 9543.82720947 M
