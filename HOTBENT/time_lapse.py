#import pygimli as pg
from pygimli.physics import ert  # the module
import numpy as np
from scipy.linalg import block_diag
import labinvesfun as inv2
from scipy import sparse
from scipy.sparse.linalg import lsqr
from scipy.sparse.linalg import lsmr
import pygimli.meshtools as mt
import matplotlib.colors as mcolors
import pygimli.physics.traveltime as tt
from pygimli.physics import TravelTimeManager
import matplotlib
import pygimli as pg
#matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import os
import gc
import timelapsefun
from joblib import Parallel, delayed

files = os.listdir('.')
ertfile = []
Data_arr = []
for i in range(len(files)):
    if files[i][-3:] == 'dat':
        ertfile.append(files[i])
        time = files[i][-17:-13] +'-'+ files[i][-13:-11] + '-' + files[i][-11:-9] + 'T' + files[i][-8:-6] + ':' + files[i][-6:-4]
        Data_arr.append(np.datetime64(time))


index = np.argsort(Data_arr)
new_Data_arr = np.sort(Data_arr)
ertfile = np.array(ertfile)
new_ertfile = ertfile[index]

name = new_ertfile
def f(nnn):
    timelapsefun.timelapsefun(nnn,name,new_Data_arr)
    gc.collect()
    
#pool = multiprocessing.Pool(processes=10)#int(multiprocessing.cpu_count())
#res = pool.map(f,range(len(name)))

#for i in range(len(name))[0:]:
#     f(i)


Parallel(n_jobs=4)(delayed(f)(i) for i in range(len(name)))

