import pygimli as pg
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
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

name = ['V2_HOTBENT2_6X1020211012_1430.dat','V2_HOTBENT2_6X1020211012_1633.dat',
        'V2_HOTBENT2_6X1020211013_0919.dat','V2_HOTBENT2_6X1020211013_1519.dat',
        'V2_HOTBENT2_6X1020211014_1430.dat','V2_HOTBENT2_6X1020211015_1430.dat',
        'V2_HOTBENT2_6X1020211018_1030.dat','V2_HOTBENT2_6X1020211021_1030.dat',
        'V2_HOTBENT2_6X1020211022_1430.dat']

f = name[1]

dataert = ert.load(f)
rhos = []
rhos.append(dataert['rhoa'].array())


## organize obs data
rhos = np.array(rhos).T
rhos_temp = rhos
rhos1 = np.log(rhos_temp)

Noid = np.load('marker.npy')
# mt.createMesh
## Mesh up and set up inital model
ert1 = ert.ERTManager(dataert)
mesh = pg.load('inv_mul1.bms')
#mesh = mesh.createH2()
# ert1.invert(mesh=mesh,lam=1)
ert1.setMesh(mesh)


rhomodel = np.median(rhos)*np.ones((mesh.cellCount(),1))

xr1 = np.log(rhomodel)
mr = xr1
mr_R = mr
mr_R = mr_R.reshape(mr_R.shape[0],1)



## forward operator
fobert = ert.ERTModelling(sr=False)
fobert.setData(dataert)
fobert.setMesh(mesh)

#---------------------data weight------------------------------------#

Delta_rhoa_rhoa = (0.03 + 0)*np.ones(rhos1[:,0].shape)
Wdert = np.diag(1.0 / np.log(Delta_rhoa_rhoa + 1))

Wd = Wdert
# ------------------------model weighting matrix--------------------- #
from pygimli.frameworks import PriorModelling
pos = [[0,0,0]]
vals = np.array([20.])
fop = PriorModelling(mesh, pos)

inv = pg.Inversion(fop=fop, verbose=False)
inv.setRegularization(cType=1)

invkw = dict(dataVals=vals, errorVals=np.ones_like(vals)*0.03, startModel=19)
inv.run(**invkw)
C = fop.constraints()

Wm_r = pg.utils.sparseMatrix2coo(C)
Wm_r = Wm_r.todense()

cw = inv.fop.regionManager().constraintWeights().array()
# cw [cw==0]=1
# Wm_r = np.diag(cw).dot(Wm_r)
Wm = Wm_r.copy()

temp = Wm.dot(Noid)
ttt = np.array(temp).reshape((-1,))
cw [abs(ttt)==1]=0
Wm =  np.diag(cw).dot(Wm_r)

# ------------------------time space weighting matrix--------------------- #
Wt = np.diagflat(np.ones(len(mr)))
# Wt[0:int(len(mr)*3/4),int(len(mr)/4):] = np.diagflat(-1*np.ones(int(len(mr)*3/4)))

def Jac(fobert, mr, mesh):
    ttt = np.reshape(mr,(-1,1),order='F')
    obs = []

    for i in range(1):
        dr, Jr = inv2.ertforandjac2(fobert, ttt[:,i])
        obs.append(dr)
        if i ==0:
            JJ = Jr
        else:
            JJ = block_diag(JJ,Jr)

    obs = np.array(obs).reshape((-1,1))
    return obs, JJ

def forward(fobert, mr, mesh):
    ttt = np.reshape(mr,(-1,1),order='F')
    obs = []

    for i in range(1):

        dr = inv2.ertforward2(fobert, ttt[:,i])
        obs.append(dr)


    obs = np.array(obs).reshape((-1,1))
    return obs


L_mr = 10#5
alpha = 0
mr = mr.reshape(mr.shape[0],1)
delta_mr = (mr-mr_R)
chi2_ert = 1
d_mr1 = np.zeros(mr.shape[0])
for nn in range(50):
    print('-------------------ERT Iteration:'+str(nn)+'---------------------------')

    dr,Jr = Jac(fobert, mr, mesh)
    dr = dr.reshape(dr.shape[0],1)
    dataerror_ert = rhos1 - dr
    fdert = (np.dot(Wd, dataerror_ert)).T.dot(np.dot(Wd,dataerror_ert))
    fmert = (L_mr*Wm.dot(mr)).T.dot(L_mr*Wm.dot(mr))
    ftert = (alpha*Wt.dot(mr)).T.dot(Wt.dot(mr))

    fc_r = fdert + fmert + ftert

    dPhi = abs(fdert/len(dr)-chi2_ert)/(fdert/len(dr))
    chi2_ert = fdert/len(dr)



    print('ERT chi2'+str(chi2_ert))
    print('ERT max error:'+str(np.max(np.abs(dataerror_ert/rhos1))))
    print('ERT mean error:'+str(np.mean(np.abs(dataerror_ert/rhos1))))
    print('ERTphi_m:'+str(fmert))
    print('dPhi:'+str(dPhi))


    if chi2_ert < 1 or dPhi < 0.01:
        break

    gc_r = Jr.T.dot(Wd.T.dot(Wd)).dot(dr-rhos1) + (L_mr*Wm).T.dot(L_mr*Wm).dot(mr) + alpha*Wt.T.dot(Wt).dot(mr)
    N11_R = (Wd.dot(Jr)).T.dot(Wd.dot(Jr)) + (L_mr*Wm).T.dot(L_mr*Wm) + alpha*Wt.T.dot(Wt)


    gc_r = np.array(gc_r)
    gc_r = gc_r.reshape((-1,))
    sN11_R = sparse.csr_matrix(N11_R)
    gc_r1 = Jr.T.dot(Wd.T.dot(Wd)).dot(dr-rhos1) + (L_mr*Wm).T.dot(L_mr*Wm).dot(mr) + alpha*Wt.T.dot(Wt).dot(mr)

    d_mr, istop, itn, normr = lsqr(sN11_R, -gc_r, damp=0.0, atol=1e-08, btol=1e-08, conlim=100000000.0,
                                   iter_lim=100, show=False, calc_var=False, x0=None)[:4]
    #d_mr, istop, itn, normr = lsmr(sN11_R, -gc_r, damp=0.0, atol=1e-06, btol=1e-06, conlim=100000000.0, maxiter=None, show=False,  x0=d_mr1)[:4]
    d_mr1 = d_mr.copy()
    d_mr = d_mr.reshape(d_mr.shape[0],1)
    mu_LS = 1.0
    iarm = 1
    while 1:
        mr1 = mr + mu_LS*d_mr

        mr1[mr1 > np.log(10000)] = np.log(10000)
        mr1[mr1 < np.log(0.0001)] = np.log(0.0001)

        dr = forward(fobert, mr1, mesh)
        dr = dr.reshape(dr.shape[0],1)

        dataerror_ert = rhos1 - dr
        fdert = (np.dot(Wd, dataerror_ert)).T.dot(np.dot(Wd,dataerror_ert))
        fmert = (L_mr*Wm.dot(mr1)).T.dot(L_mr*Wm.dot(mr1))
        ftert = (alpha*Wt.dot(mr1)).T.dot(Wt.dot(mr1))

        ft_r = fdert + fmert+ ftert

        fgoal = fc_r - 1e-4*mu_LS*(d_mr.T.dot(gc_r1.reshape(gc_r1.shape[0],1)))

        if ft_r < fgoal:
            break
        else:
            iarm = iarm+1
            mu_LS = mu_LS/2

        if iarm > 20:
            pg.boxprint('Line search FAIL EXIT')
            break

    mr = mr1
    mr[mr > np.log(10000)] = np.log(10000)
    mr[mr < np.log(0.0001)] = np.log(0.0001)

    # if L_mr > 5:
    #     L_mr = L_mr*0.7

    delta_mr = (mr-mr_R)


c = 1


mesh2 = pg.load('inv_mulbnd1.bms')
import meshop
model2 = meshop.linear_interpolation(mesh, np.array(mr[:,0]), mesh2)


mesh2['res'] = np.exp(model2)
mesh2.exportVTK(f[:-4])

