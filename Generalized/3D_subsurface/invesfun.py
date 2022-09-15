
import numpy as np
import pygimli as pg
from scipy import sparse
from scipy.sparse.linalg import lsqr


def ertforward(fob,mesh,rhomodel,xr):
    xr1 = np.log(rhomodel.array())
    xr1[mesh.cellMarkers() == 2] = np.exp(xr)

    #xr1[mesh.cellMarkers() == 1] = np.mean(np.exp(xr))
    #xr1 = np.exp(xr)
    rhomodel = pg.matrix.RVector(xr1)
    dr = fob.response(rhomodel)
    dr = np.log(dr.array())
    return dr, rhomodel

def ertforandjac(fob,rhomodel,xr):
    dr = fob.response(rhomodel)
    fob.createJacobian(rhomodel)
    J = fob.jacobian()
    J = pg.utils.gmat2numpy(J)
    J = np.exp(xr)*J
    dr = dr.array()
    J = J/dr.reshape(dr.shape[0],1)
    dr = np.log(dr)
    return dr, J

def ertforandjac2(fob,xr,mesh):
    xr1 = xr.copy()
    xr1 = np.exp(xr)
    rhomodel = pg.matrix.RVector(xr1)
    #rhomodel = xr1
    dr = fob.response(rhomodel)
    fob.createJacobian(rhomodel)
    J = fob.jacobian()
    J = pg.utils.gmat2numpy(J)
    J = np.exp(xr.T)*J
    dr = dr.array()
    J = J/dr.reshape(dr.shape[0],1)
    dr = np.log(dr)
    return dr, J

def ertforward2(fob,xr,mesh):
    xr1 = xr.copy()
    xr1 = np.exp(xr)
    rhomodel = pg.matrix.RVector(xr1)

    dr = fob.response(rhomodel)
    dr = np.log(dr)
    return dr

def srtforward(fob,sm):
    dr = fob.response(sm)
    dr = dr.array()
    return dr

def srtforandjac(fob,sm):

    dr = fob.response(sm)
    dr = dr.array()
    fob.createJacobian(sm)
    J = fob.jacobian()
    J = pg.utils.sparseMatrix2coo(J)
    J = J.todense()

    return dr, J

def srtforandjac2(fob,sm):

    dr = fob.response(np.exp(sm))
    dr = dr.array()
    fob.createJacobian(np.exp(sm))
    J = fob.jacobian()
    J = pg.utils.sparseMatrix2coo(J)
    J = J.todense()
    J = np.array(J)
    J = np.exp(sm.T)*J
    return dr, J

def srtforward2(fob,sm):
    dr = fob.response(np.exp(sm))
    dr = dr.array()
    return dr


def chi2return(L_m,w1,w2,Wm_r,Cd,J,L_cg,B,d,delta_m,mr_R,mv_R,fobert,fobsrt,mesh, rhos1,Wdert,dobs_s,Wdsrt,t):

    Cm = np.vstack((np.hstack((L_m *Wm_r,np.zeros((Wm_r.shape[0],Wm_r.shape[1]))))
                    ,np.hstack((np.zeros((Wm_r.shape[0],Wm_r.shape[1])), L_m *Wm_r))))
    N11 =np.vstack((Cd.dot(J), Cm, L_cg*B))
    R = np.vstack((Cd.dot(d), Cm.dot(delta_m), L_cg*(B.dot(delta_m) - t)))
    R = np.array(R)
    R = R.reshape((-1,))
    sN11 = sparse.csr_matrix(N11)
    delta_m, istop, itn, normr = lsqr(sN11, R, damp=0.0, atol=1e-08, btol=1e-08, conlim=100000000.0, iter_lim=None, show=False, calc_var=False, x0=None)[:4]
    delta_m = delta_m.reshape(delta_m.shape[0],1)

    mr = mr_R + delta_m[0:int(len(delta_m)/2)]
    mv = mv_R + delta_m[int(len(delta_m)/2):int(len(delta_m))]

    dr = ertforward2(fobert, mr, mesh)
    dt = srtforward2(fobsrt, mv)
    dr = dr.reshape(dr.shape[0],1)
    dt = dt.reshape(dt.shape[0],1)
    dataerror_ert = rhos1 - dr
    fdert = (np.dot(Wdert, dataerror_ert)).T.dot(np.dot(Wdert,dataerror_ert))
    chi2_ert = fdert/len(dr)

    dataerror_srt = dobs_s - dt
    fdsrt = (np.dot(Wdsrt, dataerror_srt)).T.dot(np.dot(Wdsrt, dataerror_srt))
    chi2_srt = fdsrt/len(dt)
    return chi2_ert+chi2_srt

def crossgrad(RCM,X,mr,mr_R,mv,mv_R):
    B1 = np.zeros((RCM.shape))
    B2 = np.zeros((RCM.shape))
    for i in range(np.size(RCM, 0)):
        W = np.diag(RCM[:, i])
        Xbar = np.linalg.inv((X.T.dot(W.T).dot(X))).dot((X.T.dot(W.T).dot(W)))
        Xbar1 = Xbar[0:2,:]
        g1 = Xbar1.dot(mr-mr_R)
        g2 = Xbar1.dot(mv-mv_R)

        B1[i][:] = Xbar1[0,:]*g2[1]-Xbar1[1,:]*g2[0]
        B2[i][:] = -(Xbar1[0,:]*g1[1]-Xbar1[1,:]*g1[0])

    return B1,B2


def chi2return2(L_m,w1,w2,Wm_r,Cd,J,L_cg,B,d,delta_m,mr_R,mv_R,fobert,fobsrt,mesh, rhos1,Wdert,dobs_s,Wdsrt,t):

    Cm = np.vstack((np.hstack((L_m *Wm_r,np.zeros((Wm_r.shape[0],Wm_r.shape[1]))))
                    ,np.hstack((np.zeros((Wm_r.shape[0],Wm_r.shape[1])), L_m *Wm_r))))
    N11 =np.vstack((Cd.dot(J), Cm, L_cg*B))
    R = np.vstack((Cd.dot(d), Cm.dot(np.vstack((mr_R,mv_R))), L_cg*(B.dot(delta_m) - t)))
    R = np.array(R)
    R = R.reshape((-1,))
    sN11 = sparse.csr_matrix(N11)
    delta_m, istop, itn, normr = lsqr(sN11, R, damp=0.0, atol=1e-08, btol=1e-08, conlim=100000000.0, iter_lim=None, show=False, calc_var=False, x0=None)[:4]
    delta_m = delta_m.reshape(delta_m.shape[0],1)

    mr = delta_m[0:int(len(delta_m)/2)]
    mv = delta_m[int(len(delta_m)/2):int(len(delta_m))]

    mr[mr > np.log(5000)] = np.log(5000)
    mr[mr < np.log(10)] = np.log(10)

    mv[mv > np.log(1/100)] = np.log(1/100)
    mv[mv < np.log(1/5000)] = np.log(1/5000)

    dr = ertforward2(fobert, mr, mesh)
    dt = srtforward2(fobsrt, mv)
    dr = dr.reshape(dr.shape[0],1)
    dt = dt.reshape(dt.shape[0],1)
    dataerror_ert = rhos1 - dr
    fdert = (np.dot(Wdert, dataerror_ert)).T.dot(np.dot(Wdert,dataerror_ert))
    chi2_ert = fdert/len(dr)

    dataerror_srt = dobs_s - dt
    fdsrt = (np.dot(Wdsrt, dataerror_srt)).T.dot(np.dot(Wdsrt, dataerror_srt))
    chi2_srt = fdsrt/len(dt)
    return chi2_ert+chi2_srt

# The cross gradient for spatial correlation
def crossgrad2(RCM,X,mr,mv):
    B1 = np.zeros((RCM.shape))
    B2 = np.zeros((RCM.shape))
    for i in range(np.size(RCM, 0)):
        W = np.diag(RCM[:, i])
        Xbar = np.linalg.inv((X.T.dot(W.T).dot(X))).dot((X.T.dot(W.T).dot(W)))
        Xbar1 = Xbar[0:2,:]
        g1 = Xbar1.dot(mr)
        g2 = Xbar1.dot(mv)

        B1[i][:] = Xbar1[0,:]*g2[1]-Xbar1[1,:]*g2[0]
        B2[i][:] = -(Xbar1[0,:]*g1[1]-Xbar1[1,:]*g1[0])

    return B1,B2

# The cross gradient for normal method
def crossgrad3(RCM,X,mr,mv):
    B1 = np.zeros((RCM.shape))
    B2 = np.zeros((RCM.shape))
    RCM[RCM != 0] = 1
    for i in range(np.size(RCM, 0)):

        cc = np.array(RCM[i, :])
        cc = cc.reshape((-1,))
        W = np.diag(cc)
        #Xbar = np.linalg.inv((X.T.dot(X))).dot((X.T.dot(W)))
        Xbar = np.linalg.inv((X.T.dot(W.T).dot(X))).dot((X.T.dot(W.T).dot(W)))
        Xbar1 = Xbar[0:2,:]
        g1 = Xbar1.dot(mr)
        g2 = Xbar1.dot(mv)
        g1[abs(g1)<1e-8]=0
        g2[abs(g2)<1e-8]=0
        B1[i][:] = Xbar1[0,:]*g2[1]-Xbar1[1,:]*g2[0]
        B2[i][:] = -(Xbar1[0,:]*g1[1] - Xbar1[1,:]*g1[0])
    # B1[B1<1e-8]=0
    # B2[B2<1e-8]=0
    return B1,B2
