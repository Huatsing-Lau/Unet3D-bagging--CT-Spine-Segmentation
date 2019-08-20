# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 11:23:08 2019

@author: Administrator
"""

from scipy import stats
import numpy as np

# 单样本T检验：ttest_1samp
np.random.seed(7654567)
rvs = stats.norm.rvs(loc=5,scale=10,size=(50,2))
print(stats.ttest_1samp(rvs,[1,2]))
print(stats.ttest_1samp(rvs,5.0))
print(stats.ttest_1samp(rvs,0.0))
print(stats.ttest_1samp(rvs,[5.0,0.0]))

#两独立样本t检验:ttest_ind
np.random.seed(12345678)
rvs1 = stats.norm.rvs(loc=5,scale=10,size=500)
rvs2 = stats.norm.rvs(loc=5,scale=10,size=500)
print(stats.ttest_ind(rvs1,rvs2)) #默认equal_var=True
print(stats.ttest_ind(rvs1,rvs2,equal_var=False)) #默认equal_var=True

#配对样本t检验: ttest_rel
np.random.seed(12345678)
rvs1 = stats.norm.rvs(loc=5,scale=10,size=500)
rvs2 = (stats.norm.rvs(loc=5,scale=10,size=500)+stats.norm.rvs(scale=0.2,size=500))
print(stats.ttest_rel(rvs1,rvs2))
rvs3 = (stats.norm.rvs(loc=5,scale=8,size=500)+stats.norm.rvs(scale=0.2,size=500))
print(stats.ttest_rel(rvs1,rvs3))

