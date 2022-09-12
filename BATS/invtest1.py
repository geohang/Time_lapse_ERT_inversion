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
import os
#matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

files = os.listdir('./two')
# np.datetime64('2019-12-18')

ertfile = []
Data_arr = []
for i in range(len(files)):
    if files[i][-4:] == 'Data' or files[i][-4:] == 'data':
        ertfile.append(files[i])
        files[i] = files[i][:-1]
        time = files[i][-17:-13] +'-'+ files[i][-13:-11] + '-' + files[i][-11:-9] + 'T' + files[i][-8:-6] + ':' + files[i][-6:-4] 
        Data_arr.append(np.datetime64(time))    

#f = '20191219_0146_P1.'
flag = 0
for nnn in range(len(ertfile))[-5:]:
    dataert = ert.load('./two/'+ertfile[nnn][:-4]+'dat')
    
    #dataert =ert.load('20191219_0146_P1.dat')
    rhos = []
    rhos.append(dataert['rhoa'].array())
    
    ## organize obs data
    rhos = np.array(rhos).T
    rhos_temp = rhos
    rhos1 = np.log(rhos_temp)
    
    
    ert1 = ert.ERTManager(dataert)
    mesh = pg.load('inv1.bms')
    #ert1.setMesh(mesh)
    temp = np.array(mesh.cellMarkers())
    temp = temp[temp!=1]
    
    TTT = np.array(mesh.cellMarkers())
    TTT1 = TTT[TTT!=1]
    TTT1[TTT1!=2] =2
    TTT[TTT!=1] = TTT1
    mesh.setCellMarkers(TTT)
    
    
    if flag == 0:
        fobert = ert.ERTModelling()
        fobert.setData(dataert)
        fobert.setMesh(mesh)
    else:
        fobert.setData(dataert)
        fobert.setMesh(mesh)
    
    temp2 = temp.copy()
    rhomodel = temp2.copy()
    rhomodel = np.median(rhos)*np.ones((fobert.paraDomain.cellCount(),1))
    rhomodel[temp2!=2] = 10000
    rhomodel[temp2==3] = 10000
    #rhomodel = np.median(rhos)*np.ones((fobert.paraDomain.cellCount(),1))
    
    
    
    xr1 = np.log(rhomodel)
    mr = xr1
    mr_R = mr
    mr_R = mr_R.reshape(mr_R.shape[0],1)
    
    from scipy.sparse import diags#, block_diag
    from scipy.sparse import coo_matrix
    #---------------------data weight------------------------------------#
    
    Delta_rhoa_rhoa = dataert['err'].array()  #dataert['err'].array()#(0.03 + 0)*np.ones(rhos1[:,0].shape)
    #Delta_rhoa_rhoa = (0.03 + 0)*np.ones(rhos1[:,0].shape)
    Wdert = np.diag(1.0 / np.log(Delta_rhoa_rhoa + 1))
    #Wd = coo_matrix( Wdert)
    Wd = Wdert
    # ------------------------model weighting matrix--------------------- #
    
    
    rm = fobert.regionManager()
    Ctmp = pg.matrix.RSparseMapMatrix()
    
    
    rm.setConstraintType(1)
    rm.fillConstraints(Ctmp)
    Wm_r = pg.utils.sparseMatrix2coo(Ctmp)
    
    #Wm_r = Wm_r.todense()
    
    cw = rm.constraintWeights().array()
    
    Noid = temp2.copy()
    Noid[Noid==3] = 2
    temp = Wm_r.dot(Noid)
    ttt = np.array(temp).reshape((-1,))
    cw[abs(ttt) != 0] = 0
    
    Wm_r = diags(cw).dot(Wm_r)
    
    #Wm = Wm_r.todense()
    Wm = block_diag(Wm_r.todense())
    
    def Jac(fobert, mr, mesh):
        ttt = np.reshape(mr,(-1,1),order='F')
        obs = []
    
        for i in range(1):
            dr, Jr = inv2.ertforandjac2(fobert, ttt[:,i], mesh)
            obs.append(dr)
            if i ==0:
                JJ = Jr
            else:
                JJ = block_diag((JJ,Jr))
                #JJ = block_diag(JJ,Jr)
    
        obs = np.array(obs).reshape((-1,1))
        #JJ = coo_matrix(JJ)
        return obs, JJ
    
    def forward(fobert, mr, mesh):
        ttt = np.reshape(mr,(-1,1),order='F')
        obs = []
    
        for i in range(1):
    
            dr = inv2.ertforward2(fobert, ttt[:,i], mesh)
            obs.append(dr)
    
    
        obs = np.array(obs).reshape((-1,1))
        return obs
    
    beta = 1
    L_mr = np.sqrt(300)#5;
    alpha = 25
    mr = mr.reshape(mr.shape[0],1)
    delta_mr = (mr-mr_R)
    chi2_ert = 1
    d_mr1 = np.zeros(mr.shape[0])
    
    WWd = Wd.T.dot(Wd)
    WWm_L = L_mr*Wm #.multiply()
    WWm = WWm_L.T.dot(WWm_L)
    
    Fiter = 2
    for nn in range(50):
        print('-------------------ERT Iteration:'+str(nn)+'---------------------------')
    
        dr,Jr = Jac(fobert, mr, mesh)
        dr = dr.reshape(dr.shape[0],1)
        dataerror_ert = rhos1 - dr
        
        # if nn > 3:
        #     err = np.abs((dataerror_ert)/rhos1)
        #     ind = err > 10
        #     pg.boxprint(ind)
        #     dataert['rhoa'][ind] = 0
        #     dataert.save('temp.dat')
            
        #     dataert = ert.load('temp.dat')
        #     #rhos.append(dataert['rhoa'].array())
    
        #     ## organize obs data
        #     rhos = dataert['rhoa'].array()
        #     rhos_temp = rhos.reshape(rhos.shape[0],1)
        #     rhos1 = np.log(rhos_temp)
            
        #     fobert = ert.ERTModelling()
        #     fobert.setData(dataert)
        #     fobert.setMesh(mesh)
    
        #     Delta_rhoa_rhoa = dataert['err']#(0.03 + 0)*np.ones(rhos1[:,0].shape)
       
        #     Wdert = np.diag(1.0 / np.log(Delta_rhoa_rhoa + 1))
        #     Wd = Wdert
        #     WWd = Wd.T.dot(Wd)
        #     dr,Jr = Jac(fobert, mr, mesh)
        #     dr = dr.reshape(dr.shape[0],1)
        #     dataerror_ert = rhos1 - dr
        if nn > Fiter :
            err = np.abs((dataerror_ert)/rhos1)
            
            # err = np.array(err)
            # err = err.reshape((-1,))
            Delta_rhoa_rhoa = err
            
            Delta_rhoa_rhoa = np.array(Delta_rhoa_rhoa)
            Delta_rhoa_rhoa = Delta_rhoa_rhoa.reshape((-1,))
            
            Wdert = np.diag(1.0 / np.log(Delta_rhoa_rhoa + 1))
            Wd = Wdert
            WWd = Wd.T.dot(Wd)
            Fiter = Fiter*2
        
        fdert =  (Wd.dot(dataerror_ert)).T.dot(Wd.dot(dataerror_ert))
        fmert = (WWm_L.dot(mr)).T.dot(WWm_L.dot(mr))
        #ftert = (alpha*Wt.dot(mr)).T.dot(Wt.dot(mr))
    
        fc_r = fdert  #+ fmert+ ftert
    
        dPhi = abs(fdert/len(dr)-chi2_ert)/(fdert/len(dr))
        chi2_ert = fdert/len(dr)
    
    
    
        print('ERT chi2'+str(chi2_ert))
        print('ERT max error:'+str(np.max(np.abs(dataerror_ert/rhos1))))
        print('ERT mean error:'+str(np.median(np.abs(dataerror_ert/rhos1))))
        print('ERT RMSE:'+str(np.sqrt(np.mean((dataerror_ert/rhos1)**2))))
        print('ERTphi_d:'+str(fdert), 'ERTphi_m:'+str(fmert))
        print('dPhi:'+str(dPhi))
    
    
        if chi2_ert < 1 or dPhi < 0.01:
            break
    
        gc_r = beta*Jr.T.dot(WWd).dot(dr-rhos1) + WWm.dot(mr) #+ WWt.dot(mr)
        N11_R = coo_matrix(beta*Jr.T.dot(WWd).dot(Jr)) + coo_matrix(WWm) #+ coo_matrix(WWt)
    
    
        gc_r = np.array(gc_r)
        gc_r = gc_r.reshape((-1,))
        #sN11_R = sparse.csr_matrix(N11_R)
        sN11_R = N11_R
        gc_r1 = gc_r # + alpha*Wt.T.dot(Wt).dot(mr)
    
        d_mr, istop, itn, normr = lsqr(sN11_R, -gc_r, damp=0.0, atol=1e-08, btol=1e-08, conlim=100000000.0,
                                       iter_lim=150, show=False, calc_var=False, x0=None)[:4]
        #d_mr, istop, itn, normr = lsmr(sN11_R, -gc_r, damp=0.0, atol=1e-06, btol=1e-06, conlim=100000000.0, maxiter=None, show=False,  x0=d_mr1)[:4]
        d_mr1 = d_mr.copy()
        d_mr = d_mr.reshape(d_mr.shape[0],1)
        mu_LS = 1.0
        iarm = 1
        while 1:
            mr1 = mr + mu_LS*d_mr
    
            mr1[mr1 > np.log(100000)] = np.log(100000)
            mr1[mr1 < np.log(0.00001)] = np.log(0.00001)
    
            dr = forward(fobert, mr1, mesh)
            dr = dr.reshape(dr.shape[0],1)
    
            dataerror_ert = rhos1 - dr
            fdert =  (Wd.dot(dataerror_ert)).T.dot(Wd.dot(dataerror_ert))
            fmert = (WWm_L.dot(mr1)).T.dot(WWm_L.dot(mr1))
            #ftert = (alpha*Wt.dot(mr1)).T.dot(Wt.dot(mr1))
    
            ft_r = fdert #+ fmert + ftert
    
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
    
        # ttemp = mr[Noid!=2].copy()
        
        # #ttemp[ttemp<np.max(mr[Noid==2])] = np.max(mr[Noid==2])
        # ttemp[ttemp<np.log(1000)] = np.log(1000)
        # mr[Noid!=2] = ttemp
        #mr[Noid!=2] = np.log(10000)
        #mr[temp2==3] = np.log(1000)
        if L_mr > 200:
            L_mr = L_mr*0.5
    
        delta_mr = (mr-mr_R)
    
    
    c = 1
    np.save('obs',rhos1)
    
    np.save('prd',dr)
    pcl2 = mt.readSTL('interp.stl', binary=False)
    mesh2 = mt.createMesh(pcl2,area=0.0001)
    
    import meshop
    model2 = meshop.linear_interpolation(fobert.paraDomain, mr, mesh2)
    #pg.show(mesh2,np.exp(model2))
    mesh2['res'] = np.exp(model2)
    mesh2.exportVTK('T_' + ertfile[nnn][:-4])
    del mesh2, model2
#mesh2.exportVTK('trd' + datestr
