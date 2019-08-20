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
from train_Adopted_3D_Unet import pixelAccuracy, MeanPixelAccuracy, MeanIntersectionoverUnion, DiceScore
from scipy import stats

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


def count_slice_accuracy(predicted_root,ground_truth_root,n_cls):
    slicewise_PAs = np.zeros((1,n_cls))
    slicewise_IoUs = np.zeros((1,n_cls))
    slicewise_DSs = np.zeros((1,n_cls))
    print('second observer accuracy on test dataset:')
    for (root,dirs,files) in os.walk(predicted_root):
        for file in files:
            if 'npy' in file:
                gt_seg = np.load(os.path.join(ground_truth_root,file))
                pred_seg = np.load(os.path.join(root,file)) 
                print('unique gt_seg:',np.unique(gt_seg))
                print('unique pred_seg',np.unique(pred_seg))
                for sl in range(gt_seg.shape[-1]):
                    gt_seg_slice = gt_seg[:,:,sl]
                    pred_seg_slice = pred_seg[:,:,sl]
                    if len(np.unique(gt_seg_slice))<n_cls:
                        gt_seg_slice[0][0:n_cls]=range(n_cls)
                        pred_seg_slice[0][0:n_cls]=range(n_cls)#有的sile不一定有bone/disc/nerve                                     PA = pixelAccuracy(gt_seg_slice,pred_seg_slice,n_cls=n_cls)
                    MPA,PAs = MeanPixelAccuracy(gt_seg_slice,pred_seg_slice,n_cls=n_cls)
                    MIoU, IoUs = MeanIntersectionoverUnion(gt_seg_slice,pred_seg_slice,n_cls=n_cls)
                    MDS, DSs = DiceScore(gt_seg_slice,pred_seg_slice,n_cls=n_cls)                  
                    slicewise_PAs = np.vstack((slicewise_PAs,PAs))
                    slicewise_IoUs = np.vstack((slicewise_IoUs,IoUs))
                    slicewise_DSs = np.vstack((slicewise_DSs,DSs))
                print(file+'==========')
    slicewise_PAs  = np.delete(slicewise_PAs,0,axis=0)*100
    slicewise_IoUs = np.delete(slicewise_IoUs,0,axis=0)*100
    slicewise_DSs = np.delete(slicewise_DSs,0,axis=0)*100
    
    
    return slicewise_PAs,slicewise_IoUs,slicewise_DSs


def Ttest_rel(X0,X1):
    Ttest_DSs = stats.ttest_rel(X0,X1)
    DSs0_men,DSs0_std = X0.mean(0),X0.std(0)
    DSs1_men,DSs1_std = X1.mean(0),X1.std(0)
    return Ttest_DSs,DSs0_men,DSs0_std,DSs1_men,DSs1_std


    
if __name__ == "__main__":
    n_cls = 4
    
    sec_obs_root = './data/Data_4classes_disc_testDataset_second_observer/npy/SegmentationLabel'
    ground_truth_root = './data/Data_4classes_disc_testDataset_second_observer/npy_first_observer/SegmentationLabel'
    [slicewise_PAs0,slicewise_IoUs0,slicewise_DSs0] = count_slice_accuracy(predicted_root=sec_obs_root,ground_truth_root=ground_truth_root,n_cls=n_cls)
    
    auto_predicted_root = './data/Data_4classes_disc/data_npy_20181213/5-fold/soft_ensemble_predicted_masks'#soft ensmeble
    ground_truth_root = './data/Data_4classes_disc/data_npy_20181213/test/SegmentationLabel'
    [slicewise_PAs1,slicewise_IoUs1,slicewise_DSs1] = count_slice_accuracy(predicted_root=auto_predicted_root,ground_truth_root=ground_truth_root,n_cls=n_cls)
    
    #配对样本t检验: ttest_rel
    Ttest_PAs,PAs0_men,PAs0_std,PAs1_men,PAs1_std = Ttest_rel(X0=slicewise_PAs0,X1=slicewise_PAs1)
    print(Ttest_PAs,'\n',PAs0_men,'\n',PAs0_std,'\n',PAs1_men,'\n',PAs1_std)
    
    Ttest_IoUs,IoUs0_men,IoUs0_std,IoUs1_men,IoUs1_std=Ttest_rel(X0=slicewise_IoUs0,X1=slicewise_IoUs1)
    print(Ttest_IoUs,'\n',IoUs0_men,'\n',IoUs0_std,'\n',IoUs1_men,'\n',IoUs1_std)
    
    Ttest_DSs,DSs0_men,DSs0_std,DSs1_men,DSs1_std = Ttest_rel(X0=slicewise_DSs0,X1=slicewise_DSs1)
    print(Ttest_DSs,'\n',DSs0_men,'\n',DSs0_std,'\n',DSs1_men,'\n',DSs1_std)

    # 独立样本t检验   
    Ttest_DSs=stats.ttest_ind(slicewise_DSs0,slicewise_DSs1[1:,:])
    print(Ttest_DSs)     
            