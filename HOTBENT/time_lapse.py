
import numpy as np
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
    


#for i in range(len(name))[0:]:
#    f(i)


Parallel(n_jobs=3)(delayed(f)(i) for i in range(len(name)-2))

