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
import os



files = os.listdir('./two')
ertfile = []
Data_arr = []
for i in range(len(files)):
    if files[i][-4:] == 'Data' or files[i][-4:] == 'data':
        ertfile.append(files[i])
        files[i] = files[i][:-1]
  
flagg = 0
for nnn in range(70)[1:]: #np.arange(40)[5:]:

    
    size = 3
    T1 = []
    T2 = []

    rhos = []
    dataert1 = []
    dataerr = []
    name = []
    for i in range(size):
        f = './two/'+ertfile[nnn+i][:-4]+'dat'
        name.append(ertfile[nnn+i])
        dataert = ert.load(f)
        rhos.append(dataert['rhoa'].array())
        dataert1.append(dataert)
        dataerr.append(dataert['err'].array())


    ## organize obs data
    rhos = np.array(rhos)
    rhos_temp = np.vstack((rhos[0].reshape(-1,1),rhos[1].reshape(-1,1),rhos[2].reshape(-1,1)))
    # rhos_temp = rhos[0].reshape(-1,1)
    rhos1 = np.log(rhos_temp)


    #ert1 = ert.ERTManager(dataert)
    mesh = pg.load('inv2.bms')

    temp = np.array(mesh.cellMarkers())
    temp = temp[temp!=1]

    TTT = np.array(mesh.cellMarkers())
    TTT1 = TTT[TTT!=1]
    TTT1[TTT1!=2] =2
    TTT[TTT!=1] = TTT1
    mesh.setCellMarkers(TTT)

    del TTT, TTT1
    
    if flagg == 0:
        fobert1= []
        ## forward operator
        for i in range(size):
            fobert = ert.ERTModelling()
            fobert.setData(dataert1[i])
            fobert.setMesh(mesh)
            fobert1.append(fobert)
        flagg = 1
    else:
        for i in range(size):
            #fobert = ert.ERTModelling()
            fobert1[i].setData(dataert1[i])
            #fobert.setMesh(mesh)
            #fobert1.append(fobert)        



    temp2 = temp.copy()
    #rhomodel = np.median(rhos)*np.ones((mesh1.cellCount(),1))
    rhomodeltemp = []
    for i in range(size):
        rhomodel = np.median(rhos[i])*np.ones((fobert.paraDomain.cellCount(),1))
        rhomodel[temp2!=3] = 10000
        #rhomodel[temp2==3] = 10000
        rhomodeltemp.append(rhomodel)


    xr1 = np.log(np.reshape(np.array(rhomodeltemp),(len(temp)*3,1)))
    
    del rhomodeltemp
    
    mr = xr1
    mr_R = mr
    mr_R = mr_R.reshape(mr_R.shape[0],1)




    from scipy.sparse import diags
    from scipy.sparse import coo_matrix#, block_diag
    
    #---------------------data weight------------------------------------#
    dataerr = np.array(dataerr)
    Delta_rhoa_rhoa = np.hstack((dataerr[0],dataerr[1],dataerr[2]))
   # Delta_rhoa_rhoa = dataert['err']#(0.03 + 0)*np.ones(rhos1[:,0].shape)
    
    
    Wdert = np.diag(1.0 / np.log(Delta_rhoa_rhoa + 1))
    Wd = Wdert
    del Wdert
    # ------------------------model weighting matrix--------------------- #
    rm = fobert.regionManager()
    Ctmp = pg.matrix.RSparseMapMatrix()


    rm.setConstraintType(1)
    rm.fillConstraints(Ctmp)
    Wm_r = pg.utils.sparseMatrix2coo(Ctmp)

    #Wm_r = Wm_r.todense()

    cw = rm.constraintWeights().array()

    Noid = temp2.copy()
    Noid[Noid==2] = 3
    temp = Wm_r.dot(Noid)
    ttt = np.array(temp).reshape((-1,))
    cw[abs(ttt) != 0] = 0

    Wm_r = diags(cw).dot(Wm_r)
    Wm = block_diag(Wm_r.todense(),Wm_r.todense(),Wm_r.todense())
    
    del Wm_r, cw
    # ------------------------time space weighting matrix--------------------- #
    Wt = np.zeros((int(len(mr)*2/3),len(mr)))
    # Wt = np.diagflat(np.float16(np.ones(len(mr))))
    np.fill_diagonal(Wt[:,int(len(mr)*1/3):], -1) ## adjanct time model paramters
    np.fill_diagonal(Wt, 1)

    NNoid = np.vstack((Noid.reshape(-1,1),Noid.reshape(-1,1),Noid.reshape(-1,1)))


    def Jac(fobert1, mr, mesh,size):
        ttt = np.reshape(mr,(-1,size),order='F')
        obs = []

        for i in range(size):
            dr, Jr = inv2.ertforandjac2(fobert1[i], ttt[:,i], mesh)
            obs.append(dr)
            if i ==0:
                JJ = Jr
            else:
                JJ = block_diag(JJ,Jr)

        obs = np.vstack((obs[0].reshape(-1,1),obs[1].reshape(-1,1),obs[2].reshape(-1,1)))
        return obs, JJ

    def forward(fobert1, mr, mesh, size):
        ttt = np.reshape(mr,(-1,size),order='F')
        obs = []

        for i in range(size):

            dr = inv2.ertforward2(fobert1[i], ttt[:,i], mesh)
            obs.append(dr)


        #obs = np.array(obs).reshape((-1,1))
        obs = np.vstack((obs[0].reshape(-1,1),obs[1].reshape(-1,1),obs[2].reshape(-1,1)))
        return obs

    beta = 1
    L_mr = np.sqrt(100)#5;
    alpha = np.sqrt(20)
    mr = mr.reshape(mr.shape[0],1)
    delta_mr = (mr-mr_R)
    chi2_ert = 1
    d_mr1 = np.zeros(mr.shape[0])

    # WWd = Wd.T.dot(Wd)
    # WWm_L = L_mr*Wm #.multiply()
    # WWm = WWm_L.T.dot(WWm_L)
    
    
    # del WWm_L
    t1 = 1
    t2 = 1

    coeff = np.diagflat(np.hstack((np.ones(int(len(mr)*1/size))*alpha*np.exp(-0.00*t1),
                  np.ones(int(len(mr)*1/size))*alpha*np.exp(-0.00*t2))))
                  
    Wt = coeff.dot(Wt)  
    
    del coeff
    #WWt = 
    
    for nn in range(40):
        print('-------------------ERT Iteration:'+str(nn)+'---------------------------')
        pg.boxprint('Start')
        dr,Jr = Jac(fobert1, mr, mesh,size)
        dr = dr.reshape(dr.shape[0],1)
        dataerror_ert = rhos1 - dr
        
        fdert = (Wd.dot(dataerror_ert)).T.dot(Wd.dot(dataerror_ert))
        fmert = (L_mr*Wm.dot(mr)).T.dot(L_mr*Wm.dot(mr))
        ftert = (Wt.dot(mr)).T.dot(Wt.dot(mr))

        fc_r = fdert

        dPhi = abs(fdert/len(dr)-chi2_ert)/(fdert/len(dr))
        chi2_ert = fdert/len(dr)



        print('ERT chi2'+str(chi2_ert))
        print('ERT max error:'+str(np.max(np.abs(dataerror_ert/rhos1))))
        print('ERT mean error:'+str(np.mean(np.abs(dataerror_ert/rhos1))))
        print('ERTphi_d:'+str(fdert),' ERTphi_m:'+str(fmert),' ERTphi_t:'+str(ftert))
        print('dPhi:'+str(dPhi))


        if chi2_ert < 1 or dPhi < 0.01:
            break

        gc_r = beta*Jr.T.dot(Wd.T.dot(Wd)).dot(dr-rhos1) + (L_mr*Wm).T.dot(L_mr*Wm).dot(mr) + Wt.T.dot(Wt).dot(mr)
        N11_R = coo_matrix(beta*Jr.T.dot(Wd.T.dot(Wd)).dot(Jr)) + coo_matrix((L_mr*Wm).T.dot(L_mr*Wm)) + coo_matrix(Wt.T.dot(Wt))


        gc_r = np.array(gc_r)
        gc_r = gc_r.reshape((-1,))
        #sN11_R = N11_R
        #gc_r1 = gc_r # + alpha*Wt.T.dot(Wt).dot(mr)
        
        del Jr
        d_mr, istop, itn, normr = lsqr(N11_R, -gc_r, damp=0.0, atol=1e-08, btol=1e-08, conlim=100000000.0,
                                       iter_lim=150, show=False, calc_var=False, x0=None)[:4]
        
        del N11_R, istop, itn, normr
     

        #d_mr, istop, itn, normr = lsmr(sN11_R, -gc_r, damp=0.0, atol=1e-06, btol=1e-06, conlim=100000000.0, maxiter=None, show=False,  x0=d_mr1)[:4]
        #d_mr1 = d_mr.copy()
        d_mr = d_mr.reshape(d_mr.shape[0],1)
        mu_LS = 1.0
        iarm = 1
        while 1:
            mr1 = mr + mu_LS*d_mr

            mr1[mr1 > np.log(10000)] = np.log(10000)
            mr1[mr1 < np.log(0.0001)] = np.log(0.0001)

            dr = forward(fobert1, mr1, mesh, size)
            dr = dr.reshape(dr.shape[0],1)

            dataerror_ert = rhos1 - dr
            fdert = (np.dot(Wd, dataerror_ert)).T.dot(np.dot(Wd,dataerror_ert))
            fmert = (L_mr*Wm.dot(mr1)).T.dot(L_mr*Wm.dot(mr1))
            ftert = (Wt.dot(mr1)).T.dot(Wt.dot(mr1))

            ft_r = fdert

            fgoal = fc_r - 1e-4*mu_LS*(d_mr.T.dot(gc_r.reshape(gc_r.shape[0],1)))

            if ft_r < fgoal:
                break
            else:
                iarm = iarm+1
                mu_LS = mu_LS/2

            if iarm > 20:
                pg.boxprint('Line search FAIL EXIT')
                break
        
        del d_mr, gc_r
        
        mr = mr1
        mr[mr > np.log(10000)] = np.log(10000)
        mr[mr < np.log(0.0001)] = np.log(0.0001)

        if L_mr > 200:
            L_mr = L_mr*0.5

        delta_mr = (mr-mr_R)


    c = 1

    ttt = np.reshape(mr,(-1,size),order='F')
    pcl2 = mt.readSTL('interp.stl', binary=False)
    mesh2 = mt.createMesh(pcl2,area=0.0005)
    
    import meshop
    if nnn ==0:

        model2 = meshop.linear_interpolation(fobert.paraDomain, np.array(ttt[:,0]), mesh2)
        mesh2['res'] = np.exp(model2)
        mesh2.exportVTK(name[0])

    elif nnn == len(ertfile) - 1:

        model2 = meshop.linear_interpolation(fobert.paraDomain, np.array(ttt[:,2]), mesh2)
        mesh2['res'] = np.exp(model2)
        mesh2.exportVTK(name[2])


    model2 = meshop.linear_interpolation(fobert.paraDomain, np.array(ttt[:,1]), mesh2)
    mesh2['res'] = np.exp(model2)
    mesh2.exportVTK(name[1])
    
    del mesh2
    np.save(f[:-4]+str(nnn),dr)
