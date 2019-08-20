# -*- coding: utf-8 -*-
"""
Created on Sat Feb 23 21:29:17 2019

@author: liuhuaqing
"""

import os
import numpy as np


dataset_dir = './data/Data_4classes_disc/data_npy_20181213/CT'
#'./data/Data_3classes/data_20181111/data_npy_20181111_1x1x1/CT'
Total_sl_num = 0
for (root,dirs,files) in os.walk(dataset_dir):
    for file in files:
        if 'npy' in file:
            sl_num = np.load(os.path.join(root,file)).shape[-1]
            print(file+'slicer number =',sl_num)
            Total_sl_num += sl_num
print('total slice number of dataset:',Total_sl_num)