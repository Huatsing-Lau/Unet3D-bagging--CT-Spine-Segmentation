# -*- coding: utf-8 -*-
"""
Created on Fri Jul 20 21:28:56 2018
采用训练好的3d-Uneto做目标检测
@author: liuhuaqing
"""
import os
from train_Adopted_3D_Unet import getUnet3D, pixelAccuracy, MeanPixelAccuracy, MeanIntersectionoverUnion, DiceScore
from data_3Dp import *
import argparse
import numpy as np
import matplotlib.pyplot as plt
import scipy.misc

import math
from PIL import Image

import tensorflow as tf
sess = tf.Session()


def parse_args():
    parser = argparse.ArgumentParser(description = 'parser for unet3D input arguments')
    parser.add_argument('--test_dir',default='./data/Data_4classes_disc/cross_dataset/data_npy_20181229',#'./data/Data_4classes_disc/data_npy_20181213/test',
                        help='test data set directory') # 
    parser.add_argument('--img_frame',default=32,type=int,
                        help='frame of the 3D image')
    parser.add_argument('--image_W',default=128,type=int,
                        help='width of the 3D image')
    parser.add_argument('--image_H',default=128,type=int,
                        help='height of the 3D image')
    parser.add_argument('--img_channels',default=1,type=int,
                        help='channels of the 3D image')
    parser.add_argument('--batch_size',default=1,type=int,
                        help='batch size of one train/test iteration')
    parser.add_argument('--n_cls',default=4,type=int,
                        help='num of pixel classes')
    parser.add_argument('--seg_auto_save_dir',default='./data/Data_4classes_disc/data_npy_20181213/5-fold/0/testDataset_auto_seg',
                        help='directory where the automatic predicted masks(of the test dataset) are saved')
    parser.add_argument('--checkpoint_filepath',default='./result-5-fold/5-fold-0/trained_model/train.ckpt',
                        help='checkpoint filepath of the trained model' )
    args = parser.parse_args()#保存为字典
    return args   


def readnpy2image_mask( CT_filename,label_filename ):
    image = np.load( CT_filename )+7 #-7是均值
    mask = np.load( label_filename )
        
    image = np.transpose(image,(2,0,1))#调换维度顺序后，各维度分别是：frame,width,height
    image = image[np.newaxis,:,:,:,np.newaxis]#在最后增加一个channel维度,得到的维度分别是：frame,width,height,channel
    mask = np.transpose(mask,(2,0,1))#调换维度顺序后，各维度分别是：frame,width,height,channel
    mask = mask[np.newaxis,:,:,:,np.newaxis]#在前后各增加一个维度，batch维度、channel维度
    image = (image.astype(np.float32))/1000
    mask = mask.astype(np.float32)    

    return image,mask   


def bianli(image,mask,w,h,f,n_cls):
    imageShape = image.shape
    W = imageShape[2]
    H = imageShape[3]
    F = imageShape[1]
    
    x0list = np.arange(0,W-w,w//3*2) # 形成width方向的滑窗列表
    y0list = np.arange(0,H-h,h//3*2) # 形成height方向的滑窗列表
    z0list = np.arange(0,F-f,f//2) # 形成frame方向的滑窗列表
    if not x0list[-1] == W-w:
        x0list = np.append( x0list,W-w )    # 如果不整除，则最后一个滑窗取不等距到尽头

    if not y0list[-1] == H-h:
        y0list = np.append( y0list,H-h )    # 如果不整除，则最后一个滑窗取不等距到尽头

    if not z0list[-1] == F-f:
        z0list = np.append( z0list,F-f )    # 如果不整除，则最后一个滑窗取不等距到尽头
    
    weight = np.zeros(imageShape)
    probility_mask = np.zeros( np.append(mask.shape[0:-1],n_cls) )   # 记录softmax概率值
    for x0 in x0list:
        for y0 in y0list:
            for z0 in z0list:
                image_patch = image[:,z0:z0+f,x0:x0+w,y0:y0+h,:]
                probility_mask_patch = sess.run(pred_softmax,feed_dict={inputs:image_patch})
                probility_mask[:,z0:z0+f,x0:x0+w,y0:y0+h,:] += probility_mask_patch
                weight[:,z0:z0+f,x0:x0+w,y0:y0+h,:] += 1
    output_mask = np.argmax(probility_mask/weight,4)
    return output_mask 

def make_grid(auto_masks, label_masks, nrow=5, padding=5,
              normalize=False, scale_each=False):
    """Code based on https://github.com/pytorch/vision/blob/master/torchvision/utils.py"""
    nmaps = auto_masks.shape[0]
    xmaps = min(nrow, nmaps) #水平方向坐标
    ymaps = int(math.ceil(float(nmaps) / xmaps)) #垂直方向坐标
    height, width = int(auto_masks.shape[1] + padding), int(auto_masks.shape[2] + padding) #垂直方向坐标
    grid = 2*np.ones([height * ymaps*2 + 1 + padding // 2, width * xmaps + 1 + padding // 2], dtype=np.uint8)
    k = 0
    for y in range(ymaps):
        for x in range(xmaps):
            if k >= nmaps:
                break
            h, h_width = y * height*2 + 1 + padding // 2, height - padding
            w, w_width = x * width + 1 + padding // 2, width - padding
            grid[h:h+h_width, w:w+w_width] = np.squeeze(auto_masks[k,:,:]) #预测的mask画在上方  
            h += height           
            grid[h:h+h_width, w:w+w_width] = np.squeeze(label_masks[k,:,:]) #label紧挨下方
            
            k = k + 1
    return grid

def save_image(auto_masks, label_masks, filename, nrow=5, padding=5,
               normalize=False, scale_each=False):
    ndarr = make_grid(auto_masks, label_masks, nrow=nrow, padding=padding,
                            normalize=normalize, scale_each=scale_each)
    plt.figure(figsize=[50,100])
    plt.imshow( ndarr.astype(int) )
    plt.axis('off')
    plt.title('Automatic Mask vs. Manual Mask')
    plt.savefig(filename)
    plt.show()
    #im = Image.fromarray( np.float32(ndarr)/2.0*255.0 )
    #im.save(filename)  

def middle_cut(img,h,w):
    H,W = np.shape(img)
    x,y = int((H-h)/2),int((W-w)/2)
    cut_img = img[x:x+h,y:y+w]
    return cut_img
     

if __name__ == "__main__":
    
    args = parse_args()
    test_dir = args.test_dir
    img_frame = args.img_frame
    img_channels = args.img_channels
    image_W = args.image_W #输入（输出）图像宽度方向的像素个数
    image_H = args.image_H #输入（输出）图像高度方向的像素个数
    batch_size = args.batch_size #一个批量图像中的图像数量
    n_cls = args.n_cls  #类别总数,包括背景
    checkpoint_filepath = args.checkpoint_filepath
    seg_auto_save_dir = args.seg_auto_save_dir
    
    if not os.path.exists(seg_auto_save_dir):
        os.makedirs(seg_auto_save_dir)

    inputs, preds = getUnet3D(img_frame,image_W,image_H,img_channels,n_cls)#定义神经网络的结构
    pred_softmax = tf.nn.softmax(preds)#运算之后pred_softmax于preds的维度一样
    # mask取值方式：比较几个通道的预测值（表示概率），取最大值所在的通道
    pred_mask = tf.expand_dims(tf.cast(tf.argmax(pred_softmax,4),tf.float32),dim=-1)#argmax按照最大值所在层，降维度,输出的shape是(?, 256, 256),所以要用expand_dims恢复维度
    labels = tf.placeholder(tf.int32, shape=(None,img_frame,image_W,image_H,img_channels))
    
    saver = tf.train.Saver()
    init_op = tf.global_variables_initializer()
    
    
    CT_files = os.listdir(os.path.join(test_dir,'CT'))
    label_files = os.listdir(os.path.join(test_dir,'SegmentationLabel'))
    
    with sess.as_default():
        
        sess.run( init_op )
        saver.restore(sess,checkpoint_filepath)      
        
        result_save_path = test_dir
        h,w = 270,180#test数据集则取300,180；cross dataset则取270,180
        label_masks = np.ones([len(CT_files),h,w])
        auto_masks = np.ones([len(CT_files),h,w])

        k = 0
        N = len(CT_files)
        plt.figure()
        plt.axis('off'),plt.tight_layout(pad=0,h_pad=0,w_pad=0)
        

        PAs_testdataset=[]
        IoUs_testdataset=[]
        DSs_testdataset=[]
        for CT_filename,label_filename in zip( CT_files, label_files ):
            print('label_filename:',label_filename)
            image,mask = readnpy2image_mask( os.path.join(test_dir,'CT',CT_filename),
                                            os.path.join(test_dir,'SegmentationLabel',label_filename) )
            start_time = time.time()
            output_mask = bianli(image,mask,image_W,image_H,img_frame,n_cls)
            stop_time = time.time()
            print('bianli time:',stop_time-start_time)
            trueMask = np.array(mask).flatten().astype(int)
            predMask = np.array(output_mask).flatten().astype(int)
            
            PA = pixelAccuracy(trueMask,predMask,n_cls)
            MPA,PAs = MeanPixelAccuracy(trueMask,predMask,n_cls)
            MIoU, IoUs = MeanIntersectionoverUnion(trueMask,predMask,n_cls)
            MDS, DSs = DiceScore(trueMask,predMask,n_cls)
            
            print('PA=',PA*100)
            print('MPA=',MPA*100,'PAs=',PAs*100)
            print('MIoU=',MIoU*100,'IoUs=',IoUs*100)
            print('Mean Dice Score=',MDS*100,'Dice Scores=',DSs*100)

            PAs_testdataset.append(PAs)
            IoUs_testdataset.append(IoUs)
            DSs_testdataset.append(DSs)
            
            print(np.shape(label_masks))
            label_masks_k = np.squeeze(mask[0,20,:,:,0])
            print(np.shape(label_masks_k))
                
            sqz = np.squeeze(mask[0,20,:,:,0])
            print('sqz:',np.shape(sqz))
            tmp = middle_cut( np.squeeze(mask[0,20,:,:,0]),h,w )
            print('mask:',np.shape(mask))
            print('tmp:',np.shape(tmp))
            label_masks[k,:,:] = middle_cut( np.squeeze(mask[0,20,:,:,0]),h,w ) #变为3D数组
            auto_masks[k,:,:] = middle_cut( np.squeeze(output_mask[0,20,:,:]),h,w ) #变为3D数组

            plt.subplot(2,N,k+1),plt.imshow( np.squeeze(mask[0,20,:,:,0].astype(int)) ),plt.axis('off')
            plt.subplot(2,N,k+1+N),plt.imshow( np.squeeze(output_mask[0,20,:,:].astype(int)) ),plt.axis('off')

            fn_saveto = os.path.join(seg_auto_save_dir,'seg_auto_'+label_filename)
            np.save(fn_saveto,np.squeeze(output_mask,0))#
            k += 1
            
        plt.show()

        print('label_masks:',np.shape(label_masks))
        print('auto_masks:',np.shape(auto_masks))
        filename = '5fold-0'    
        x_path = os.path.join(result_save_path, '{}_{}.png'.format(filename, 24))
        save_image(auto_masks, label_masks, x_path, nrow=10, padding=2)
        
        meanPAs = np.mean(np.array(PAs_testdataset),0)
        meanIoUs = np.mean(np.array(IoUs_testdataset),0)
        meanDSs = np.mean(np.array(DSs_testdataset),0)
        print('meanPAs:',meanPAs)
        print('meanIoUs',meanIoUs)
        print('meanDSs',meanDSs)
    
    
    
    
    
    
    
