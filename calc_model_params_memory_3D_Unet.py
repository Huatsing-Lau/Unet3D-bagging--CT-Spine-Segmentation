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
from keras.models import *
from keras.layers import Input, Concatenate, Activation, Conv3D, Conv3DTranspose, MaxPooling3D, UpSampling3D, Dropout, Cropping3D
from keras.layers.normalization import BatchNormalization as BN

#model = applications.VGG16(input_shape=(1024,1024,3),include_top=False,weights=None)
#print("无全连接层总参数量:",model.count_params())

def getUnet3D(img_frame,img_rows,img_cols,img_channels,n_cls):
    
    input_shape = (img_frame,img_rows,img_cols,img_channels)#input_shape = (img_frame,img_rows,img_cols,img_channels)
    inputs = Input(input_shape)
    print(inputs)
    
    conv1 = Conv3D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
    conv1 = BN(axis=-1)(conv1)
    print("conv1 shape:",conv1.shape)
    conv1 = Conv3D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
    conv1 = BN(axis=-1)(conv1)
    print("conv1 shape:",conv1.shape)
    pool1 = MaxPooling3D(pool_size=(2, 2, 2))(conv1)
    print("pool1 shape:",pool1.shape)
    
    conv2 = Conv3D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
    print("conv2 shape:",conv2.shape)
    conv2 = Conv3D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
    conv2 = BN(axis=-1)(conv2)
    print("conv2 shape:",conv2.shape)
    pool2 = MaxPooling3D(pool_size=(2, 2, 2))(conv2)
    print("pool2 shape:",pool2.shape)
    
    conv3 = Conv3D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
    print("conv3 shape:",conv3.shape)
    conv3 = Conv3D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
    conv3 = BN(axis=-1)(conv3)
    print("conv3 shape:",conv3.shape)
    pool3 = MaxPooling3D(pool_size=(2, 2, 2))(conv3)
    print("pool3 shape:",pool3.shape)
    
    conv4 = Conv3D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
    conv4 = Conv3D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
    conv4 = BN(axis=-1)(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling3D(pool_size=(2, 2, 2))(drop4)
    
    conv5 = Conv3D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
    conv5 = Conv3D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
    conv5 = BN(axis=-1)(conv5)
    print("conv5 shape:",conv5.shape)
    drop5 = Dropout(0.5)(conv5)
    print("drop5 shape:",drop5.shape)
            
    up6 = Conv3D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling3D(size = (2,2,2))(drop5))
    print("up6 shape:",up6.shape)
    merge6 = Concatenate(axis=4)([drop4,up6])
    #merge6 = merge([drop4,up6], mode = 'concat', concat_axis = 4)
    #merge6 = merge.Concatenate(axis = 4)([drop4,up6])
    conv6 = Conv3D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
    conv6 = Conv3D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)
    conv6 = BN(axis=-1)(conv6)
    
    up7 = Conv3D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling3D(size = (2,2,2))(conv6))
    merge7 = Concatenate(axis=4)([conv3,up7])
    #merge7 = merge([conv3,up7], mode = 'concat', concat_axis = 4)
    #merge7 = merge.Concatenate(axis = 4)([conv3,up7])
    conv7 = Conv3D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
    conv7 = Conv3D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)
    conv7 = BN(axis=-1)(conv7)
    
    up8 = Conv3D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling3D(size = (2,2,2))(conv7))
    merge8 = Concatenate(axis=4)([conv2,up8])
    #merge8 = merge([conv2,up8], mode = 'concat', concat_axis = 4)
    #merge8 = merge.Concatenate(axis = 4)([conv2,up8])
    conv8 = Conv3D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
    conv8 = Conv3D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)
    conv8 = BN(axis=-1)(conv8)
    
    up9 = Conv3D(32, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling3D(size = (2,2,2))(conv8))
    merge9 = Concatenate(axis=4)([conv1,up9])
    #merge9 = merge([conv1,up9], mode = 'concat', concat_axis = 4)
    #merge9 = merge.Concatenate(axis = 4)([conv1,up9])
    conv9 = Conv3D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
    conv9 = Conv3D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv9 = BN(axis=-1)(conv9)
    preds = Conv3D(n_cls, 1, kernel_initializer = 'he_normal')(conv9)
    print("preds shape:",preds.shape)

    model = Model(input=inputs,output=preds)
    return model



n_cls = 3 #像素类别总数（含背景）
img_frame = 32
img_channels = 1
img_rows = 64
img_cols = 64

    
model = getUnet3D(img_frame,img_rows,img_cols,img_channels,n_cls)
all_params_memory = 0
all_feature_memory = 0
for layer in model.layers:
    #训练权重w占用的内存
    params_memory = layer.count_params()/(1024*1024) * 4 #单位转化为MB
    print("训练权重w占用的内存:",layer.name,layer.count_params(),str(params_memory)+" M")
    all_params_memory = all_params_memory + params_memory
    #特征图占用内存
    feature_shape = layer.output_shape
    feature_size = 1
    for i in range(1,len(feature_shape)):
        feature_size = feature_size*feature_shape[i]
    feature_memory = feature_size/(1024*1024) * 4 #单位转化为MB
    print(layer.name)
    print("特征图占用内存:",feature_shape,feature_size,str(feature_memory)+" M")
    all_feature_memory = all_feature_memory + feature_memory

    
print("网络权重W占用总内存:",str(all_params_memory)+" M")
print("网络特征图占用总内存:",str(all_feature_memory)+" M")
print("网络总消耗内存:",str(all_params_memory+all_feature_memory)+" M")
# 网络权重W占用总内存: 8327.79214477539 M
# 网络特征图占用总内存: 1216.0350647 M
# 网络总消耗内存: 9543.82720947 M
