# -*- coding: utf-8 -*-
"""
Created on Thu Feb 21 08:43:30 2019
采用cross validation ensemble的方法统计模型在测试集上的准确率

功能：
第二位观察者实验：
6个测试病例的分割mask对比图；
总体的混淆矩阵、归一化矩阵；
每个测试病例的Dice、PA、IoU、Sensitivity、Precision；
@author: Administrator
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from keras.utils import to_categorical
from train_Adopted_3D_Unet import pixelAccuracy, MeanPixelAccuracy, MeanIntersectionoverUnion, DiceScore


def plot_confusion_matrix(cm,labels,title='Confusion Matrix',cmap=plt.cm.binary,dtype='int'):
    plt.figure(figsize=(4.5,3),dpi=120)
    ind_array = np.arange(len(labels))
    x,y = np.meshgrid(ind_array,ind_array)
    for x_val,y_val in zip(x.flatten(),y.flatten()):
        c = cm[y_val][x_val]
        if dtype=='int':
            plt.text(x_val,y_val,int(c),color='black',fontsize=8,va='center',ha='center')
        else:
            plt.text(x_val,y_val,'%0.3f' %(c,),color='black',fontsize=8,va='center',ha='center')
    plt.imshow(cm,interpolation='nearest',cmap=cmap)
    plt.title(title,fontsize=10)
    plt.colorbar()
    xlocations = np.array(range(len(labels)))
    plt.xticks(xlocations,labels)#rotation=90
    plt.yticks(xlocations,labels)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')


sec_obs_root = './data/Data_4classes_disc_testDataset_second_observer/npy/SegmentationLabel'
ground_truth_root = './data/Data_4classes_disc_testDataset_second_observer/npy_first_observer/SegmentationLabel'
#'./data/Data_4classes_disc/data_npy_20181213/test/SegmentationLabel'





y_true = np.array([])
y_predicted = np.array([])
print('second observer accuracy on test dataset:')
for (root,dirs,files) in os.walk(sec_obs_root):
    for file in files:
        if 'npy' in file:
            gt_seg = np.load(os.path.join(ground_truth_root,file))
            sec_obs_seg = np.load(os.path.join(root,file)) 
            
            
            print('unique gt_seg:',np.unique(gt_seg))
            print('unique sec_obs_seg',np.unique(sec_obs_seg))
            
            PA = pixelAccuracy(gt_seg,sec_obs_seg,n_cls=4)
            MPA,PAs = MeanPixelAccuracy(gt_seg,sec_obs_seg,n_cls=4)
            MIoU, IoUs = MeanIntersectionoverUnion(gt_seg,sec_obs_seg,n_cls=4)
            MDS, DSs = DiceScore(gt_seg,sec_obs_seg,n_cls=4)
            
            print(file+'==========')
            print('PA:',PAs*100)
            print('IoU:',IoUs*100)
            print('Dice Score:',DSs*100)
            
#            print(np.where( (gt_seg.flatten()==sec_obs_seg.flatten())==False))
            
            y_true = np.append(y_true,gt_seg.flatten())
            y_predicted = np.append(y_predicted,sec_obs_seg.flatten())
            assert y_true.shape==y_predicted.shape
            
    cm = confusion_matrix(y_true,y_predicted)
    print('cm=')
    print(cm)
    labels = ['background','bone','disc','nerve']
    #np.set_printoptions(precision=3)
    cn_normalized = cm.astype('float')/cm.sum(axis=1)
    print('cn_normalized=')
    print(cn_normalized)
            
    plot_confusion_matrix(cm.astype(int),labels,title='Confusion Matrix',cmap=plt.cm.spring,dtype='int')#cmap=plt.cm.binary#plt.cm.spring
    plt.show()
    plot_confusion_matrix(cn_normalized,labels,title='Normalized Confusion Matrix',cmap=plt.cm.spring,dtype='float')#cmap=plt.cm.binary#plt.cm.spring
    plt.show()
            
            