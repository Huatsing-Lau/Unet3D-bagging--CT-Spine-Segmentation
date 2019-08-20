# -*- coding: utf-8 -*-
"""
Created on Sat May 19 17:13:47 2018
# tensorflow 读取指定路径下的npy格式数据文件，生成自己的训练/测试数据(3D，width/height尺寸不计)
# 服务于老范发我的神经根数据
@author: liuhuaqing
"""
import time
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
 

def read_npy_file(item1,item2,frame,width,height,patch_or_not=True):
    image = np.load(item1)+7 #-70.75是均值
    mask = np.load(item2)

    if patch_or_not:
        imageShape = image.shape
        x1 = np.random.randint(low=0, high=imageShape[0]-width, dtype='l')#这个方法产生离散均匀分布的整数，这些整数大于等于low，小于high。
        y1 = np.random.randint(low=0, high=imageShape[1]-height, dtype='l')
        z1 = np.random.randint(low=0, high=imageShape[2]-frame, dtype='l')
        image = image[x1:x1+width,y1:y1+height,z1:z1+frame]
        mask = mask[x1:x1+width,y1:y1+height,z1:z1+frame]          
    
    image = np.transpose(image,(2,0,1))#调换维度顺序后，各维度分别是：frame,width,height
    image = image[:,:,:,np.newaxis]#在最后增加一个channel维度,得到的维度分别是：frame,width,height,channel
    mask = np.transpose(mask,(2,0,1))#调换维度顺序后，各维度分别是：frame,width,height,channel
    mask = mask[:,:,:,np.newaxis]#在最后增加一个channel维度
    
    noise = np.random.normal(0, 5, image.shape) #正太分布随机噪声
    image = image+noise
    image = (image.astype(np.float32))/1000
    mask = mask.astype(np.float32)    
    
    if np.random.rand()>0.5:
        if np.random.rand()>0.5:
            image = np.flip(image,2) #上下翻转
            mask = np.flip(mask,2)
        else:
            image = np.flip(image,3) #左右翻转
            mask = np.flip(mask,3)                      
        
    return (image,mask)



def get_files(file_dir):
    # step1；获取file_dir下所有的图路径名
    image_list = []
    mask_list = []
    for file in os.listdir(file_dir+'/CT'):
        image_list.append(file_dir+'/CT'+'/'+file)
    for file in os.listdir(file_dir+'/SegmentationLabel'):
        mask_list.append(file_dir+'/SegmentationLabel'+'/'+file)
        
    #step2: 对生成的图片路径和标签做打乱处理
    #利用shuffle打乱顺序
    temp = np.array([image_list,mask_list])
    temp = temp.transpose()#n行2列
    np.random.shuffle(temp)
    #从打乱顺序的temp中取list
    image_list = list(temp[:,0])#打乱顺序的文件路径名字符串
    mask_list = list(temp[:,1])#打乱顺序的文件路径名字符串
    return image_list , mask_list




def get_dataBatch(data_dir,D,W,H,batch_size,patch_or_not=True):
    image_list , mask_list = get_files(data_dir)
    assert len(image_list)==len(mask_list)
    num = len(image_list)
    idxs = np.random.randint(0,num,batch_size) # 随机数
    
    if patch_or_not:
        image_batch = np.zeros([batch_size,D,W,H,1])
        mask_batch = np.zeros([batch_size,D,W,H,1])
        i = 0
        for k in idxs:
            image,mask = read_npy_file(image_list[k],mask_list[k],D,W,H,patch_or_not=True)
            image_batch[i,:,:,:,:] = image
            mask_batch[i,:,:,:,:] = mask
            i += 1
        image_mask_batch = (image_batch,mask_batch)
    else:
        batch_size = 1
        idxs = np.random.randint(0,num,batch_size) # 随机数
        image,mask = read_npy_file(image_list[idxs[0]],mask_list[idxs[0]],D,W,H,patch_or_not=False)
        image = image[np.newaxis,:,:,:,:]
        mask = mask[np.newaxis,:,:,:,:]
        image_mask_batch = (image,mask)
    return image_mask_batch


# 以下代码是为了测试以上函数正确性，读取指定路径的图片，加工后，在控制台显示出来
if __name__ == "__main__":
    batch_size = 4
    train_dir = './data/Data_3classes/data_20181111/data_npy_20181111/test' 

    
    tt = time.time()
    for i in range(20):
        try:
            print(i)
            image_mask_batch = get_dataBatch(train_dir,2,100,100,batch_size,patch_or_not=False)
            imgs = image_mask_batch[0]
            masks = image_mask_batch[1]
            plt.figure()
            plt.subplot(1,2,1),plt.imshow( imgs[0,1,:,:,0] ),plt.axis('off'),plt.title('CT')
            plt.subplot(1,2,2),plt.imshow( masks[0,1,:,:,0]/2.0*255.0 ),plt.axis('off'),plt.title('label')
            plt.show()
        except tf.errors.OutOfRangeError:
            print('start new epoch')
        print(time.time() - tt)
        
        
