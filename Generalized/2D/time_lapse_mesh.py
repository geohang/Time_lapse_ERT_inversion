import pygimli as pg
from pygimli.physics import ert  # the module
import numpy as np
from scipy.linalg import block_diag
import invesfun as inv2
from scipy import sparse
from scipy.sparse.linalg import lsqr

import pygimli.meshtools as mt

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

rhos = []
for i in range(4):


    dataert = ert.load("Tert"+str(i+1))
    rhos.append(dataert['rhoa'].array())


## organize obs data
rhos = np.array(rhos)
rhos_temp = rhos.T.reshape(rhos.shape[0]*rhos.shape[1],1,order='F')
rhos1 = np.log(rhos_temp)


## Mesh up and set up inital model
ert1 = ert.ERTManager(dataert)

geo = pg.meshtools.createParaMeshPLC(dataert,quality=32,paraDX=0.5, paraMaxCellSize=4,
                                     boundaryMaxCellSize=3000,smooth=[2, 2], balanceDepth=True)

line1 = mt.createPolygon([[-20,-2],[20,-2]], isClosed=False,
                          interpolate='linear')

line2 = mt.createPolygon([[-20,-8],[20,-8]], isClosed=False,
                          interpolate='linear')
geo1 = geo + line2 +line1


grid = pg.meshtools.createMesh(geo1, quality=32)
grid.save('inv.bms')
c=1
