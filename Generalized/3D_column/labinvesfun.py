
import numpy as np
import pygimli as pg
from scipy import sparse
from scipy.sparse.linalg import lsqr



def ertforandjac2(fob,xr):

    xr1 = np.exp(xr)

    rhomodel = xr1
    dr = fob.response(rhomodel)
    fob.createJacobian(rhomodel)
    J = fob.jacobian()
    J = pg.utils.gmat2numpy(J)
    J = np.exp(xr.T)*J
    dr = dr.array()
    J = J/dr.reshape(dr.shape[0],1)
    dr = np.log(dr)
    return dr, J

def ertforward2(fob,xr):

    xr1= np.exp(xr)
    rhomodel = pg.matrix.RVector(xr1)

    dr = fob.response(rhomodel)
    dr = np.log(dr)
    return dr
