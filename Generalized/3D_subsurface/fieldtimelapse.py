import pygimli as pg
from pygimli.physics import ert  # the module
import numpy as np
from scipy.linalg import block_diag
import invesfun as inv2
from scipy import sparse
from scipy.sparse.linalg import lsqr
from scipy.sparse.linalg import lsmr
import pygimli.meshtools as mt
import matplotlib.colors as mcolors
import pygimli.physics.traveltime as tt
from pygimli.physics import TravelTimeManager
import matplotlib
#matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import timelapseinvfun
import gc
import multiprocessing

import os
files = os.listdir('./two')
# np.datetime64('2019-12-18')

ertfile = []
Data_arr = []
for i in range(len(files)):
    if files[i][-4:] == 'Data' or files[i][-4:] == 'data':
        ertfile.append(files[i])
        files[i] = files[i][:-1]
  
        
    

tttimes = range(60)

for ww in range(60)[4:]:
    timelapseinvfun.timelapsefun(ww,ertfile)
    gc.collect()
  