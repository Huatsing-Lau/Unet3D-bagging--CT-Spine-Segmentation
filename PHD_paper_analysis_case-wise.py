# -*- coding: utf-8 -*-
"""
Created on Thu Feb 21 08:43:30 2019
统计模型在测试集上的各种准确率指标
@author: Administrator
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix,precision_score,recall_score,f1_score
from keras.utils import to_categorical
from train_Adopted_3D_Unet import pixelAccuracy, MeanPixelAccuracy, MeanIntersectionoverUnion, DiceScore
from scipy import stats
from test_Adopted_3D_Unet import middle_cut,save_image

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


def count_case_accuracy(pred_root,ground_truth_root,n_cls):
    # 统计数据集中每个测试病例的Dice、PA、IoU、Sensitivity、Precision
    casewise_PAs = np.zeros((1,n_cls))
    casewise_IoUs = np.zeros((1,n_cls))
    casewise_DSs = np.zeros((1,n_cls))
    casewise_Sensitivitys = np.zeros((1,n_cls))
    casewise_Precisions = np.zeros((1,n_cls))
    print('second observer accuracy on test dataset:')
    for (root,dirs,files) in os.walk(pred_root):
        for file in files:
            if 'npy' in file:
                gt_seg = np.load(os.path.join(ground_truth_root,file))
                pred_seg = np.load(os.path.join(root,file)) 
                pred_seg = np.transpose(pred_seg,(1,2,0))#SPINET2需要执行该语句，SPINET3-4不需要
                print('unique gt_seg:',np.unique(gt_seg))
                print('unique pred_seg',np.unique(pred_seg))

                MPA,PAs = MeanPixelAccuracy(gt_seg,pred_seg,n_cls=n_cls)
                MIoU, IoUs = MeanIntersectionoverUnion(gt_seg,pred_seg,n_cls=n_cls)
                MDS, DSs = DiceScore(gt_seg,pred_seg,n_cls=n_cls) 
                Precisions = precision_score(y_true=gt_seg.flatten(), y_pred=pred_seg.flatten(), average=None)
                Sensitivitys = recall_score(y_true=gt_seg.flatten(), y_pred=pred_seg.flatten(), average=None) 
                
                casewise_PAs = np.vstack((casewise_PAs,PAs))
                casewise_IoUs = np.vstack((casewise_IoUs,IoUs))
                casewise_DSs = np.vstack((casewise_DSs,DSs))
                casewise_Sensitivitys = np.vstack((casewise_Sensitivitys,Sensitivitys))
                casewise_Precisions = np.vstack((casewise_Precisions,Precisions))
                print(file+'==========')
    casewise_PAs  = np.delete(casewise_PAs,0,axis=0)*100
    casewise_IoUs = np.delete(casewise_IoUs,0,axis=0)*100
    casewise_DSs = np.delete(casewise_DSs,0,axis=0)*100
    casewise_Sensitivitys  = np.delete(casewise_Sensitivitys,0,axis=0)
    casewise_Precisions  = np.delete(casewise_Precisions,0,axis=0)

    return casewise_PAs,casewise_IoUs,casewise_DSs,casewise_Sensitivitys,casewise_Precisions


def Ttest_rel(X0,X1):
    Ttest_DSs = stats.ttest_rel(X0,X1)
    DSs0_men,DSs0_std = X0.mean(0),X0.std(0)
    DSs1_men,DSs1_std = X1.mean(0),X1.std(0)
    return Ttest_DSs,DSs0_men,DSs0_std,DSs1_men,DSs1_std


def flatten_dataset(pred_root,ground_truth_root,n_cls):
    y_true = np.array([])
    y_pred = np.array([])
    print('accuracy on test dataset:')
    for (root,dirs,files) in os.walk(pred_root):
        for file in files:
            if 'npy' in file:
                gt_seg = np.load(os.path.join(ground_truth_root,file))
                pred_seg = np.load(os.path.join(root,file)) 
                pred_seg = np.transpose(pred_seg,(1,2,0))#SPINET2需要执行该语句，SPINET3-4不需要
                print(file+'==========')
                y_true = np.append(y_true,gt_seg.flatten())
                y_pred = np.append(y_pred,pred_seg.flatten())
                assert y_true.shape==y_pred.shape
    return y_true.astype(int), y_pred.astype(int)



def draw_mask_compared_figure(pred_root,ground_truth_root,n_cls,h,w,savefile_name):
    #绘制y_pred与y_ture的对比图   
    print('mask comparision on test dataset:')
    k = 0
    num = 0
    for (root,dirs,files) in os.walk(pred_root):
        for file in files:
            if 'npy' in file:
                num+=1
        y_ture = np.ones([num,h,w])
        y_pred = np.ones([num,h,w])        
        for file in files:        
            if 'npy' in file:
                gt_seg = np.load(os.path.join(ground_truth_root,file))
                pred_seg = np.load(os.path.join(root,file))
                pred_seg = np.transpose(pred_seg,(1,2,0))#SPINET2需要执行该语句，SPINET3-4不需要
                y_ture[k,:,:] = middle_cut( np.squeeze(gt_seg[:,:,20]),h,w ) #变为3D数组
                y_pred[k,:,:] = middle_cut( np.squeeze(pred_seg[:,:,20]),h,w ) #变为3D数组
                k += 1
    save_image(y_pred, y_ture, savefile_name, nrow=k+1, padding=2)

   
    
if __name__ == "__main__":
    n_cls = 4
    labels = ['background','bone','disc','nerve']
    
    auto_pred_root = './data/Data_4classes_disc/data_npy_20181213/5-fold/4/testDataset_auto_seg'
    #'./data/Data_4classes_disc/data_npy_20181213/5-fold/0/testDataset_auto_seg' #本数据集 SPINET2
    #'./data/Data_4classes_disc/data_npy_20181213/5-fold/hard_ensemble_predicted_masks' #本数据集硬集成 SPINET3
    #'./data/Data_4classes_disc/data_npy_20181213/5-fold/soft_ensemble_predicted_masks'#本数据集soft ensmeble SPINET4
    #'./data/Data_4classes_disc/cross_dataset/data_npy_20190313/soft_ensemble_predicted_masks' #跨数据集自动分割 soft ensmeble（SPINET4）
    #'./data/Data_4classes_disc/data_npy_20181213/5-fold/soft_ensemble_predicted_masks'#本数据测试集soft ensmeble SPINET4
    #'./data/Data_4classes_disc/cross_dataset/data_npy_20190313_selected6/soft_ensemble_predicted_masks' #跨数据集自动分割 soft ensmeble selected6（SPINET4）
    gt_root = './data/Data_4classes_disc/data_npy_20181213/test/SegmentationLabel' #本数据测试集真值
    #'./data/Data_4classes_disc/data_npy_20181213/test/SegmentationLabel' 本数据测试集真值
    #'./data/Data_4classes_disc/cross_dataset/data_npy_20190313/SegmentationLabel' #跨数据测试集真值
    [casewise_PAs1,casewise_IoUs1,casewise_DSs1,casewise_Sensitivitys1,casewise_Precisions1] = count_case_accuracy(pred_root=auto_pred_root,ground_truth_root=gt_root,n_cls=n_cls)
    
# =============================================================================
#     # 换Mask对比图
#     h,w = 270,180
#     savefile_name = os.path.join(auto_pred_root, 'hard_ensemble_comparision_mask.png')
#     draw_mask_compared_figure(pred_root=auto_pred_root,ground_truth_root=gt_root,n_cls=n_cls,h=h,w=w,savefile_name=savefile_name) 
# =============================================================================
      
    # 统计soft ensemble混淆矩阵
    y_true, y_pred = flatten_dataset(pred_root=auto_pred_root,ground_truth_root=gt_root,n_cls=n_cls)
#    cm = confusion_matrix(y_true,y_pred)
#    plot_confusion_matrix(cm.astype(int),labels=labels,title='Confusion Matrix',cmap=plt.cm.spring,dtype='int')#cmap=plt.cm.binary#plt.cm.spring
#    plt.show()
#    cn_normalized = cm.astype('float')/cm.sum(axis=1)
#    plot_confusion_matrix(cn_normalized,labels,title='Normalized Confusion Matrix',cmap=plt.cm.spring,dtype='float')#cmap=plt.cm.binary#plt.cm.spring
#    plt.show()
    
    #
    MPA,PAs = MeanPixelAccuracy(y_true,y_pred,n_cls=n_cls)
    
    # 数据集上的准确率和敏感度
#    dataset_Precisions_auto_pred = precision_score(y_true=y_true, y_pred=y_pred, average=None)
#    dataset_Sensitivitys_auto_pred = recall_score(y_true=y_true, y_pred=y_pred, average=None)
    dataset_f1scores_auto_pred = f1_score(y_true=y_true, y_pred=y_pred, average=None)
    dataset_macro_f1scores_auto_pred = f1_score(y_true=y_true, y_pred=y_pred, average='macro')

    
    
        
            