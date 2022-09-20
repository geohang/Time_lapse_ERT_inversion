import numpy as np
import timelapseinvfun
import gc
from joblib import Parallel, delayed
#import matplotlib
#matplotlib.use('TkAgg')

import os
files = os.listdir('./two')

para_flag = 1 # flag for parallel computing
Nums_cpu = 3 # the number of used cpu
Lambda = 10 # the regularization parameter for the model smooth parameter
alpha = 2 # the regularization parameter for the time space smooth parameter
decay_rate = -0.01 # the decay rate for the non-uniform time space
widsize = 3 # the window size for window time lapse inversion

######################################################################################
# The below part is for ERT file name organized
# You should change the below part and make sure you file name is a numpy array and in
# right time order

ertfile = []
Data_arr = []
for i in range(len(files)):
    if files[i][-3:] == 'dat':
        ertfile.append(files[i])
        time = files[i][-17:-13] +'-'+ files[i][-13:-11] + '-' + files[i][-11:-9] + 'T' + files[i][-8:-6] + ':' + files[i][-6:-4]
        Data_arr.append(np.datetime64(time))
        
index = np.argsort(Data_arr)
new_Date_arr = np.sort(Data_arr)
ertfile = np.array(ertfile)
new_ertfile = ertfile[index]

index = np.argsort(Data_arr)
new_Date_arr = np.sort(Data_arr)
ertfile = np.array(ertfile)
new_ertfile = ertfile[index]
###########################################################################################

name = new_ertfile  # the file name array for ERT data file in right time order
Date = new_Date_arr # the time array for corresponding ERT data file, in np.datetime64 format


def f(nnn):
    timelapseinvfun.timelapsefun(nnn,name,Date,size=widsize,Lambda=Lambda,alpha=alpha,decay_rate=decay_rate)
    gc.collect()





if para_flag == 0:
    for i in range(len(name)-int(widsize/2)):
       f(i)
else:
    Parallel(n_jobs=Nums_cpu)(delayed(f)(i) for i in range(len(name)-int(widsize/2)))
