
"""
# 这个是和老范一起做的神经根分割项目
# 读取的4类别数据，'background':0,'bone':1,'disc':2,'nerve':3 共4类标签
# 一共有31例，其中6例为test集，25例为五折交叉训练验证集
"""

import os
import PIL.Image as Image

from datetime import datetime
import time
import argparse

#import tensorflow as tf
#sess = tf.Session()

import tensorflow as tf
#import tensorlayer as tl
import os

# 参考：https://blog.csdn.net/leibaojiangjun1/article/details/53671257
os.environ["CUDA_VISIBLE_DEVICES"] = '0,1'   #指定第一块GPU可用
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 1.0  # 程序最多只能占用指定gpu100%的显存
config.gpu_options.allow_growth = True      #程序按需申请内存
sess = tf.Session(config = config)


import numpy as np
from keras.models import *
from keras.layers import Input, Concatenate, Activation, Conv3D, Conv3DTranspose, MaxPooling3D, UpSampling3D, Dropout, Cropping3D
from keras.layers.normalization import BatchNormalization as BN
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.utils import multi_gpu_model
from data_3Dp import *

from keras import backend as keras
keras.set_session(sess)

  
# 函数count_num_model_params()的作用：统计模型参数量  
def count_num_model_params():
    print( np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()]) ) 

    
# 函数weighted_softmax_cross_entropy_loss的功能是：定义加权交叉熵损失函数
def weighted_softmax_cross_entropy_loss(logits, labels):
    weights = [1,1,1,1]
    n_cls = 4
    with tf.name_scope('loss'):

        logits = tf.reshape(logits, [-1, tf.shape(logits)[4]], name='flatten_logits')
        labels = tf.reshape(labels, [-1], name='flatten_labels')

        weight_map = tf.to_float(tf.equal(labels, 0, name='label_map_0')) * weights[0]
        for i in range(1,n_cls):
            weight_map = weight_map + tf.to_float(tf.equal(labels, i, name='label_map_'+ str(i))) * weights[i]
        # 若采用迭代，则会报错：TypeError: 'Tensor' object is not iterable.  因此暂且采用以上笨方法
        #for i, weight in enumerate(weights[1:], start=1):
            #weight_map = weight_map + tf.to_float(tf.equal(labels, i, name='label_map_' + str(i))) * weight

        # 权重矩阵，该矩阵大小是batch_size*image_W*image_H
        weight_map = tf.stop_gradient(weight_map, name='stop_gradient')

        # 计算交叉熵损失矩阵，该矩阵大小是batch_size*image_W*image_H，与像素一一对应
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits, name='cross_entropy_softmax')
        
        # 计算加权交叉熵损失矩阵，该矩阵大小是batch_size*image_W*image_H，加权交叉熵损失矩阵 = 交叉熵损失矩阵与交叉熵损失矩阵相同位置的元素相乘
        weighted_cross_entropy = tf.multiply(weight_map, cross_entropy, name='apply_weights')

        # 计算交叉熵损失函数值，交叉熵损失函数值=交叉熵损失矩阵的所有元素的平均值
        loss = tf.reduce_mean(weighted_cross_entropy, name='loss')

    #return loss, weight_map
    return loss



# 函数getUnet2D的功能是定义U-net的结构

def getUnet3D(img_frame,img_rows,img_cols,img_channels,n_cls):
    
    input_shape = (img_frame,None,None,img_channels)#input_shape = (img_frame,img_rows,img_cols,img_channels)
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


def parse_args():
    parser = argparse.ArgumentParser(description = 'parser for unet3D input arguments')
    parser.add_argument('--n_cls',default=4,
                        help='total number of pixel classes')
    parser.add_argument('--train_dir',default='./data/Data_4classes_disc/data_npy_20181213/5-fold/0/training', 
                        help='train data directory')
    parser.add_argument('--validation_dir',default='./data/Data_4classes_disc/data_npy_20181213/5-fold/0/validation',
                        help='validation data directory')
    parser.add_argument('--img_frame',default=32,type=int,
                        help='frame of the 3D image')
    parser.add_argument('--image_W',default=64,type=int,
                        help='width of the 3D image')
    parser.add_argument('--image_H',default=64,type=int,
                        help='height of the 3D image')
    parser.add_argument('--img_channels',default=1,type=int,
                        help='channels of the 3D image')
    parser.add_argument('--batch_size',default=4,type=int,
                        help='batch size of one train/test iteration')
    parser.add_argument('--outputMask_save_dir',default='./outputMask',
                        help='directory where the predicted masks(test) are saved')
    parser.add_argument('--weights',default=[1,2,20,20],#'background':0,'bone':1,'disc':2,'nerve':3
                        help='A list of the weights associated with the different labels in the ground truth.')
    parser.add_argument('--learning_rate',default=0.0005,
                        help='learning rate')
    parser.add_argument('--training_iters',default=50000,type=int,#50000
                        help='total iteration times of the training')
    parser.add_argument('--validation_iter',default=6,type=int,help='number of cases to be evaluated in a single validation')
    parser.add_argument('--display_step',default=250,type=int,
                        help='dispaly training loss/accuracy every display_step times of training iteration')
    
    args = parser.parse_args()#保存为字典
    return args


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



# 语义分割准确率的定义和计算，参考：https://blog.csdn.net/majinlei121/article/details/78965435
def fast_hist(a, b, n):
    k = (a >= 0) & (a < n) #正常情况下全是True
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n**2).reshape(n, n)#np.bincount 用于统计数组中（从小到大）给取值出现的次数

def Hist(a,b,n):
    hist = fast_hist(a,b,n)
    return hist
    
def pixelAccuracy(trueMask,predMask,n_cls):
    hist = Hist(trueMask,predMask,n_cls)
    PA = np.diag(hist).sum() / hist.sum()
    return PA

def MeanPixelAccuracy(trueMask,predMask,n_cls):
    epsilon = 1
    hist = Hist(trueMask,predMask,n_cls)
    PAs = np.diag(hist) / hist.sum(1)
    MPA = np.nanmean(PAs)
    return MPA, PAs

def MeanIntersectionoverUnion(trueMask,predMask,n_cls):
    epsilon = 1
    hist = Hist(trueMask,predMask,n_cls)
    IoUs = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))
    MIoU = np.nanmean(IoUs)
    return MIoU, IoUs

def DiceScore(trueMask,predMask,n_cls):
    epsilon = 1
    hist = Hist(trueMask,predMask,n_cls)
    correct_pred = np.diag(hist) # 给类别正确预测的像素点数
    pred_classes = np.sum(hist,0) # 预测处的各类别像素点数,
    true_classes = np.sum(hist,1) # 真实的各类别像素点数
    DSs = 2*correct_pred/(pred_classes+true_classes)
    MDS = np.nanmean(DSs)
    return MDS, DSs


if __name__ == "__main__":
    args = parse_args()
    n_cls = args.n_cls #像素类别总数（含背景）
    img_frame = args.img_frame
    img_channels = args.img_channels
    image_W = args.image_W #输入（输出）图像宽度方向的像素个数
    image_H = args.image_H #输入（输出）图像高度方向的像素个数
    batch_size = args.batch_size #一个批量图像中的图像数量
    train_dir = args.train_dir # 训练数据的路径
    validation_dir = args.validation_dir
    ouputMask_save_dir = args.outputMask_save_dir
    record_dstdir = './trainning_record_unet_3Dp'
    initial_weights = args.weights#训练期间给几类像素的loss权值
    learning_rate = args.learning_rate
    training_iters = args.training_iters#200000#30000 #训练阶段的总迭代次数#若执行训练，这将该值修改为非零整数，若想单单执行预测（测试集），则将该值修改为0
    validation_iter = args.validation_iter
    display_step = args.display_step#20训练期间，每迭代display_step次，就在控制台上显示一次训练结果（accuracy等参数）

    capacity = 12
    bestAccuracy = 0 #0.867#0.849#
    bestLost = 10 #0.29#0.163#
    gpus = 1

    MIoU_record = []
    IoUs_record = []
    MDS_record = []
    DSs_record = []
    loss_record = []
    IterationStep_record = []
       
    if gpus<=1:
        print('[INFO] training with 1 GPU...')
        model = getUnet3D(img_frame,image_W,image_H,img_channels,n_cls)#定义神经网络的结构
    else:
        print('[INFO] training with {} GPUs...'.format(gpus))
        with tf.device('/cpu:0'):
            model1 = getUnet3D(img_frame,image_W,image_H,img_channels,n_cls)#定义神经网络的结构
            model = multi_gpu_model(model1,gpus=gpus)
# =============================================================================
#     labels = tf.placeholder(tf.int32, shape=(None,img_frame,image_W,image_H,img_channels))#tf.int32
#     weights = tf.placeholder(tf.float32,shape=(n_cls,))
#     loss = weighted_softmax_cross_entropy_loss(model.outputs,labels)#定义损失函数
# 
#     #labels = tf.placeholder(tf.float32, shape=(None,img_frame,image_W,image_H,img_channels))
#     #loss = 1 - tl.cost.dice_coe(preds,labels)  # dice coefficient损失函数
#     
#     optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, 
#                                        beta1=0.9, beta2=0.999, 
#                                        epsilon=1e-08, 
#                                        use_locking=False, 
#                                        name='Adam').minimize(loss)#定义求解器       
# =============================================================================
    
    
    model.compile(optimizer=Adam(lr=learning_rate), loss='categorical_crossentropy')#weighted_softmax_cross_entropy_loss
    
    
    #unet预测出来的mask
    #print('preds shape:',preds.shape)
    pred_softmax = tf.nn.softmax(model.output)#运算之后pred_softmax于preds的维度一样
    # mask取值方式：比较6个通道的预测值（表示概率），取最大值所在的通道，如某像素的6个通道的预测值分别是(0.1,0.2,0.1,0.2,0.4),可见第6个通道的预测值最大，故该像素属于L5,则令对应位置的mask为5
    pred_mask = tf.expand_dims(tf.cast(tf.argmax(pred_softmax,4),tf.float32),dim=-1)#argmax按照最大值所在层，降维度,输出的shape是(?, 256, 256),所以要用expand_dims恢复维度
    print('pred_mask shape',pred_mask.shape)
    pred_mask2 = model.output_mask
    
    saver = tf.train.Saver()
    checkpoint_filepath = r'trained_model/train.ckpt'#
    
    init_op = tf.global_variables_initializer()
    
    count_num_model_params()
    with sess.as_default():
        
        sess.run([init_op])
        #saver.restore(sess,checkpoint_filepath)
        step = 0
        
        # 下面这个while循环的作用是训练，总迭代次数为training_iters，每一次循环迭代的次数是batch_size
        start_time = time.time()
        print('strat training at time: ', start_time)
        while step*batch_size < training_iters:#not coord.should_stop() and 
            try:
                step += 1
                weightsValue = initial_weights if step*batch_size<=training_iters*2/5 else [1,1,2]
                image_mask_batch = get_dataBatch(train_dir,img_frame,image_W,image_H,batch_size,patch_or_not=True)# 读取一批训练数据
                imgs =  image_mask_batch[0]
                masks = image_mask_batch[1]
                print('masks shape:',masks.shape)
                print('imgs shape:',masks.shape)
                #print(batch_size,img_frame,image_W,image_H,img_channels)
                assert imgs.shape == (batch_size,img_frame,image_W,image_H,img_channels)
                assert masks.shape == (batch_size,img_frame,image_W,image_H,img_channels)
                
                model.fit(imgs,masks)

                _,loss_value = sess.run([optimizer,loss],feed_dict={inputs:imgs, labels:masks, weights:weightsValue})#训练
                
                step_duration = time.time() - start_time
                print('%s, Step %s, Iter %s,loss= %s, Duration: %s sec, done!'
                      %(datetime.now(),
                        step,
                        step*batch_size,
                        loss_value,
                        np.float('{:.2f}'.format(step_duration)) ) )
                
                # validation:
                if step%display_step == 0:#每display_step显示一次训练过程中的loss和accuracy,执行一次validation
                    valset_PA = []
                    valset_MPA = []
                    valset_PAs = []
                    valset_MIoU = []
                    valset_IoUs = []
                    valset_MDS = []
                    valset_DSs = []                    
                    for val_iter in range(validation_iter):
                        image_mask_batch = get_dataBatch(validation_dir,img_frame,image_W,image_H,batch_size=1,patch_or_not=False)#读取验证数据
                        image =  image_mask_batch[0]
                        mask = image_mask_batch[1]
                        output_mask = bianli(image,mask,image_W,image_H,img_frame,n_cls)
                    
                        trueMask = np.array(mask).flatten().astype(int)
                        predMask = np.array(output_mask).flatten().astype(int)
                        PA = pixelAccuracy(trueMask,predMask,n_cls)
                        MPA,PAs = MeanPixelAccuracy(trueMask,predMask,n_cls)
                        MIoU,IoUs = MeanIntersectionoverUnion(trueMask,predMask,n_cls)
                        MDS, DSs = DiceScore(trueMask,predMask,n_cls)
                        
                        valset_PA.append(PA)
                        valset_MPA.append(MPA)
                        valset_PAs.append(PAs)
                        valset_MIoU.append(MIoU)
                        valset_IoUs.append(IoUs)
                        valset_MDS.append(MDS)
                        valset_DSs.append(DSs)
                    
                    MIoU_record.append( np.mean(valset_MIoU,0) )
                    IoUs_record.append( np.mean(valset_IoUs,0) )
                    MDS_record.append( np.mean(valset_MDS,0) )
                    DSs_record.append( np.mean(valset_DSs,0) )
                    loss_record.append( loss_value )
                    IterationStep_record.append(step)
                    
                    # 命令行窗口显示
                    print('validation:  %s, Step %s, Iter %s, PA= %s, MPA= %s, PAs=%s, MIoU= %s, IoUs=%s, MDS=%s, DSs=%s'
                          %(datetime.now(),
                            step,step*batch_size,
                            np.float('{:.6f}'.format(np.mean(valset_PA,0)) ),
                            np.float('{:.6f}'.format(np.mean(valset_MPA)) ),
                            np.mean(valset_PAs,0),
                            np.float('{:.6f}'.format( MIoU_record[-1]) ),
                            IoUs_record[-1],
                            np.float('{:.6f}'.format( MDS_record[-1]) ),
                            DSs_record[-1],
                            ) 
                          )
                    
                    
                    MMDS = np.nanmean(MDS_record[-2:-1]) if step>10 else np.nanmean(MDS_record[0:-1])
                    Mloss = np.nanmean(loss_record[-2:-1]) if step>10 else np.nanmean(loss_record[0:-1])
                    print(MMDS,Mloss)
                    if MMDS > bestAccuracy: # and Mloss < bestLost:
                        saver.save(sess,checkpoint_filepath)
                        bestAccuracy = MMDS
                        bestLost = Mloss
                        print('best NetWork refreshed!')
                    print('bestAccuracy = ',bestAccuracy,'bestLost=',bestLost)
                    
                    np.save(os.path.join(record_dstdir,'MIoU_record.npy'),MIoU_record)
                    np.save(os.path.join(record_dstdir,'IoUs_record.npy'),IoUs_record)
                    np.save(os.path.join(record_dstdir,'MDS_record.npy'),MDS_record)
                    np.save(os.path.join(record_dstdir,'DSs_record.npy'),DSs_record)
                    np.save(os.path.join(record_dstdir,'loss_record.npy'),loss_record)
                    np.save(os.path.join(record_dstdir,'IterationStep_record.npy'),IterationStep_record)
                    print('Accuracy Record Sucessfully Saved to *.npy files')

                print(step)
            except tf.errors.OutOfRangeError:
                print('here is "except tf.errors.OutOfRangeError"!')
        
        print('Training done!!!')
        stop_time = time.time()
        print('Training time comsumed: ',(stop_time - start_time) ,'s')
               

        sess.close()      
    



