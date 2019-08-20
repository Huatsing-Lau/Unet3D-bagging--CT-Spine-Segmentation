# -*- coding: utf-8 -*-
"""
Created on Fri Sep 28 21:46:37 2018
绘制训练过程的曲线图
@author: liuhuaqing
"""
import numpy as np
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.pyplot import savefig


root = './result-5-fold/5-fold-0/trainning_record_unet_3Dp'
DS_file = os.path.join(root,'DSs_record.npy')
PAs_file = os.path.join(root,'PAs_record.npy')
loss_file = os.path.join(root,'loss_record.npy')
iter_file = os.path.join(root,'IterationStep_record.npy')

DSs = 100*np.load(DS_file)  #unit:%
PAs = 100*np.load(PAs_file)  #unit:%
loss = np.load(loss_file)
steps = np.load(iter_file)

print('steps_max',np.max(steps))

batch_size = 4
iteration = steps*batch_size

Epoch = (iteration/(20*3*2*2))



#这里导入你自己的数据
#......
#......
#x_axix，train_pn_dis这些都是长度相同的list()

#开始画图=========================================================
#sub_axix = filter(lambda x:x%200 == 0, y)
figsave_root = root

# 画DSs图
plt.figure()
#plt.title('Dice Scores  of training period')
plt.plot(Epoch, DSs[:,0], color='green', label='background')
plt.plot(Epoch, DSs[:,1], color='blue', label='bone')
plt.plot(Epoch, DSs[:,2], color='orange', label='vessel')
plt.plot(Epoch, DSs[:,3], color='red', label='nerve')
plt.xlim(0,int(np.max(Epoch)))
plt.legend() # 显示图例
plt.xlabel('Epoch')
plt.ylabel('Dice Score/(%)')
savefig(os.path.join(figsave_root,'DSs.png'))
plt.show()


# 画PAs图
plt.figure()
#plt.title('Dice Scores of training period')
plt.plot(Epoch, PAs[:,0], color='green', label='background')
plt.plot(Epoch, PAs[:,1], color='blue', label='bone')
plt.plot(Epoch, PAs[:,2], color='orange', label='vessel')
plt.plot(Epoch, PAs[:,3], color='red', label='nerve')
plt.xlim(0,int(np.max(Epoch)))
plt.legend() # 显示图例
plt.xlabel('Epoch')
plt.ylabel('Pixel Accuracy/(%)')
savefig(os.path.join(figsave_root,'PAs.png'))
plt.show()




plt.figure()
plt.plot(Epoch, loss, color='red')
plt.xlim(0,int(np.max(Epoch)))
plt.xlabel('Epoch')
plt.ylabel('Loss')
savefig(os.path.join(figsave_root,'Loss.png'))
plt.show()

# 参考：本文来自 Site1997 的CSDN 博客 ，全文地址请点击：https://blog.csdn.net/site1997/article/details/79180384?utm_source=copy 
