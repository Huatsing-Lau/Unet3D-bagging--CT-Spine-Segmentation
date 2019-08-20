# -*- coding: utf-8 -*-
"""
Created on Thu Feb 21 08:43:30 2019
采用cross validation ensemble的方法统计模型在测试集上的准确率
@author: Administrator
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from keras.utils import to_categorical
from train_Adopted_3D_Unet import getUnet3D, pixelAccuracy, MeanPixelAccuracy, MeanIntersectionoverUnion, DiceScore


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
    


def ensemble(files,mode='hard'):
    if mode=='soft':
        auto_seg_probability = np.load(files[0]) 
        for file in files[1:]:
            k_auto_seg_probability = np.load(file)
            auto_seg_probability += k_auto_seg_probability
        ensemble_auto_seg = auto_seg_probability.argmax(axis=3)
    if mode=='hard':
        auto_seg_one_hot = to_categorical(np.load(files[0]))
        for file in files[1:]:
            k_auto_seg = np.load(file)
            k_one_hot = to_categorical(k_auto_seg)
            auto_seg_one_hot += k_one_hot
        ensemble_auto_seg = auto_seg_one_hot.argmax(axis=3)
    return ensemble_auto_seg
 



               
def ensemble_cm_dataset(saveDir,five_fold_root,fold_01234,labels,mode='hard'):
    y_true = np.array([])
    y_predicted = np.array([])
    print('cross validation soft emsemble:')
    for (root,dirs,files) in os.walk(os.path.join(five_fold_root,fold_01234[0])):
        for file in files:
            if 'npy' in file:
                files = []
                for k,fold_k in enumerate(fold_01234):
                    files.append( os.path.join(five_fold_root,fold_k,file) )
                    
                ensemble_auto_seg = ensemble(files,mode=mode)
                ensemble_auto_seg = np.transpose(ensemble_auto_seg,(1,2,0))# 顺序是 HWD
                
                np.save(os.path.join(saveDir,file),ensemble_auto_seg)

                gt_file = file.split('_')[-1]
                gt_seg = np.load(os.path.join(ground_truth_root,gt_file))
                
            
                MPA,PAs = MeanPixelAccuracy(gt_seg,ensemble_auto_seg,n_cls=4)
                MIoU, IoUs = MeanIntersectionoverUnion(gt_seg,ensemble_auto_seg,n_cls=4)
                MDS, DSs = DiceScore(gt_seg,ensemble_auto_seg,n_cls=4)
            
                print(gt_file+'==========')
                print('PA:',PAs*100)
                print('IoU:',IoUs*100)
                print('Dice Score:',DSs*100)
            
                y_true = np.append(y_true,gt_seg.flatten())
                y_predicted = np.append(y_predicted,ensemble_auto_seg.flatten())
    
    cm = confusion_matrix(y_true,y_predicted)
    np.set_printoptions(precision=3)
    cn_normalized = cm.astype('float')/cm.sum(axis=1)       
    plot_confusion_matrix(cm.astype(int),labels,title='Confusion Matrix',cmap=plt.cm.spring,dtype='int')#cmap=plt.cm.binary#plt.cm.spring
    plt.show()
    plot_confusion_matrix(cn_normalized,labels,title='Normalized Confusion Matrix',cmap=plt.cm.spring,dtype='float')#cmap=plt.cm.binary#plt.cm.spring
    plt.show()
    
    return cm,cn_normalized


# 跨数据集   软集成 
# =============================================================================
# five_fold_root = './data/Data_4classes_disc/cross_dataset/data_npy_20190313'
# fold_01234 = ['autoseg_probability_5-fold-0','autoseg_probability_5-fold-1','autoseg_probability_5-fold-2','autoseg_probability_5-fold-3','autoseg_probability_5-fold-4']
# ground_truth_root = './data/Data_4classes_disc/cross_dataset/data_npy_20190313/SegmentationLabel' 
# saveDir = './data/Data_4classes_disc/cross_dataset/data_npy_20190313/soft_ensemble_predicted_masks'
# =============================================================================

# =============================================================================
# # 本数据集 软集成
# five_fold_root = './data/Data_4classes_disc/data_npy_20181213/5-fold'
# fold_01234 = ['0/testDataset_auto_seg_probability','1/testDataset_auto_seg_probability','2/testDataset_auto_seg_probability','3/testDataset_auto_seg_probability','4/testDataset_auto_seg_probability']
# ground_truth_root = './data/Data_4classes_disc/data_npy_20181213/test/SegmentationLabel' 
# saveDir = './data/Data_4classes_disc/data_npy_20181213/5-fold/soft_ensemble_predicted_masks'
# =============================================================================


# =============================================================================
# # 跨数据集   硬集成 
# five_fold_root = './data/Data_4classes_disc/cross_dataset/data_npy_20181229'
# fold_0 = 'autoseg_5-fold-0'
# fold_1234 = ['autoseg_5-fold-1','autoseg_5-fold-2','autoseg_5-fold-3','autoseg_5-fold-4']
# ground_truth_root = './data/Data_4classes_disc/cross_dataset/data_npy_20181229/SegmentationLabel' 
# =============================================================================


# 本数据集 硬集成
five_fold_root = './data/Data_4classes_disc/data_npy_20181213/5-fold'
fold_01234 = ['0/testDataset_auto_seg','1/testDataset_auto_seg','2/testDataset_auto_seg','3/testDataset_auto_seg','4/testDataset_auto_seg']
ground_truth_root = './data/Data_4classes_disc/data_npy_20181213/test/SegmentationLabel'
saveDir = './data/Data_4classes_disc/data_npy_20181213/5-fold/hard_ensemble_predicted_masks'

labels = ['background','bone','disc','nerve']
if not os.path.exists(saveDir):
    os.makedirs(saveDir)

    
cm,cn_normalized = ensemble_cm_dataset(saveDir,five_fold_root,fold_01234,labels,mode='hard')

