from pygimli.physics import ert  # the module
import numpy as np
from scipy.linalg import block_diag
import labinvesfun as inv2
from scipy.sparse.linalg import lsqr
import pygimli as pg
import gc



#### This function is mainly for the window time lapse ERT inversion

def timelapsefun(nnn,name,new_Data_arr):
    size = 3

    rhos = []
    dataert1 = []
    dataerr = []
    for i in range(size):
        f = name[i+nnn]
        dataert = ert.load(f)
        rhos.append(dataert['rhoa'].array())
        dataert1.append(dataert)
        dataerr.append(dataert['err'].array())

    ## organize obs data
    rhos = np.array(rhos)
    rhos_temp = rhos[0]
    for i in range(size-1):
        rhos_temp = np.hstack((rhos_temp,rhos[i+1]))

        
    rhos_temp = np.array(rhos_temp)
    rhos_temp = rhos_temp.reshape((-1,1))
    rhos1 = np.log(rhos_temp)



    ## Mesh up and set up inital model
    ert1 = ert.ERTManager(dataert)

    ## load mesh, change here for your own mesh
    mesh = pg.load('inv_mul1.bms')
    ert1.setMesh(mesh)



    fobert1= []
    ## forward operator
    for i in range(size):
        fobert = ert.ERTModelling(sr=False)
        fobert.setData(dataert1[i])
        fobert.setMesh(mesh)
        fobert1.append(fobert)

    ### this part is for the structure constrain
    Noid = np.load('marker.npy')
    
    temp2 = Noid.copy()
    #rhomodel = np.median(rhos)*np.ones((mesh1.cellCount(),1))
    rhomodeltemp = []
    for i in range(size):
        rhomodel = np.median(rhos[i])*np.ones((mesh.cellCount(),1))
        rhomodeltemp.append(rhomodel)
        
    xr1 = np.log(np.reshape(np.array(rhomodeltemp),(len(temp2)*size,1)))
    
    del rhomodeltemp
    
    mr = xr1
    mr_R = mr
    mr_R = mr_R.reshape(mr_R.shape[0],1)        
        
    from scipy.sparse import diags
    from scipy.sparse import coo_matrix#, block_diag
    #---------------------data weight------------------------------------#
    dataerr = np.array(dataerr)
    Delta_rhoa_rhoa = np.hstack((dataerr[0],dataerr[1],dataerr[2]))
    #Delta_rhoa_rhoa = (0.10 + 0)*np.ones(rhos1[:,0].shape)
    Wdert = np.diag(1.0 / np.log(Delta_rhoa_rhoa + 1))

    Wd = Wdert
    #Wd = coo_matrix(Wd)

    # ------------------------model weighting matrix--------------------- #
    rm = fobert.regionManager()
    Ctmp = pg.matrix.RSparseMapMatrix()


    rm.setConstraintType(1)
    rm.fillConstraints(Ctmp)
    Wm_r = pg.utils.sparseMatrix2coo(Ctmp)
    Wm_r = Wm_r.todense()
    #Wm_r = Wm_r.todense()

    cw = rm.constraintWeights().array()
    Wm = Wm_r.copy()
    

    
    temp = Wm.dot(Noid)
    ttt = np.array(temp).reshape((-1,))
    
    cw[abs(ttt) == 2] = 1 - np.exp(-0.1*nnn)
    cw[abs(ttt) == 1] = 1 - np.exp(-0.1*nnn)
      
    Wm = np.diag(cw).dot(Wm_r)
    del Wm_r, Wdert
    gc.collect()
    #Wm = coo_matrix(Wm)

    #Wm = block_diag((Wm,Wm,Wm))
    Wm = block_diag(Wm,Wm,Wm)

    # ------------------------time space weighting matrix--------------------- #
    Wt = np.zeros((int(len(mr)*2/3),len(mr)))
    # Wt = np.diagflat(np.float16(np.ones(len(mr))))
    np.fill_diagonal(Wt[:,int(len(mr)*1/3):], -1) ## adjanct time model paramters
    np.fill_diagonal(Wt, 1)

    NNoid = np.vstack((Noid.reshape(-1,1),Noid.reshape(-1,1),Noid.reshape(-1,1)))
    #Wt = coo_matrix(Wt)

    def Jac(fobert1, mr, size):
        ttt = np.reshape(mr,(-1,size),order='F')
        obs = []

        for i in range(size):
            dr, Jr = inv2.ertforandjac2(fobert1[i], ttt[:,i])
            obs.append(dr)
            #Jr = Jr
            if i ==0:
                JJ = Jr
            else:
                #JJ = block_diag((JJ,Jr))
                JJ = block_diag(JJ,Jr)
        #obs = np.array(obs).T
        #JJ = coo_matrix(JJ)
        obs = np.vstack((obs[0].reshape(-1,1),obs[1].reshape(-1,1),obs[2].reshape(-1,1)))
        return obs, JJ

    def forward(fobert1, mr, size):
        ttt = np.reshape(mr,(-1,size),order='F')
        obs = []

        for i in range(size):

            dr = inv2.ertforward2(fobert1[i], ttt[:,i])
            obs.append(dr)


        #obs = np.array(obs).T
        obs = np.vstack((obs[0].reshape(-1,1),obs[1].reshape(-1,1),obs[2].reshape(-1,1)))
        return obs


    L_mr = 5#5;
    alpha = 1 #5
    mr = mr.reshape(mr.shape[0],1)
    delta_mr = (mr-mr_R)
    chi2_ert = 1
    d_mr1 = np.zeros(mr.shape[0])
    #TT = [3,16,6,20,24,3*24,3*24,24]

    time_d = new_Data_arr[nnn+1] - new_Data_arr[nnn]
    t1 = time_d.data[0]/60

    time_d = new_Data_arr[nnn+2] - new_Data_arr[nnn+1]
    t2 = time_d.data[0]/60
    if t1>24:
        t1 = 24

    
    if t2>24:
        t2 = 24


        
        
        
    coeff = np.diagflat(np.hstack((np.ones(int(len(mr)*1/size))*alpha*np.exp(-0.01*t1),
                  np.ones(int(len(mr)*1/size))*alpha*np.exp(-0.01*t2))))

    Wt = coeff.dot(Wt)
    WWd = Wd.T.dot(Wd)

    #WWm = (Wm.multiply(L_mr)).T.dot(Wm.multiply(L_mr))

    WWm = (L_mr*Wm).T.dot(L_mr*Wm)
    WWt = Wt.T.dot(Wt)

    for nn in range(15):
        print('-------------------ERT Iteration:'+str(nn)+'---------------------------')

        dr,Jr = Jac(fobert1, mr, size)
        dr = dr.reshape(dr.shape[0],1)
        dataerror_ert = rhos1 - dr
        fdert = (Wd.dot(dataerror_ert)).T.dot(Wd.dot(dataerror_ert))
        fmert = (L_mr*Wm.dot(mr)).T.dot(L_mr*Wm.dot(mr))
        ftert = (Wt.dot(mr)).T.dot(Wt.dot(mr))

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

        gc_r = Jr.T.dot(WWd).dot(dr-rhos1) + WWm.dot(mr) + WWt.dot(mr)
        N11_R = coo_matrix(Jr.T.dot(WWd).dot(Jr)) + coo_matrix(WWm) + coo_matrix(WWt)


        gc_r = np.array(gc_r)
        gc_r = gc_r.reshape((-1,))
        #sN11_R = sparse.csr_matrix(N11_R)
        sN11_R = N11_R
        gc_r1 = gc_r#Jr.T.dot(Wd.T.dot(Wd)).dot(dr-rhos1) + (Wm.multiply(L_mr)).T.dot(Wm.multiply(L_mr)).dot(mr) + alpha*Wt.T.dot(Wt).dot(mr)
        
        del Jr
        gc.collect()
        d_mr, istop, itn, normr = lsqr(sN11_R, -gc_r, damp=0.0, atol=1e-08, btol=1e-08, conlim=100000000.0,
                                       iter_lim=150, show=False, calc_var=False, x0=None)[:4]
        #d_mr, istop, itn, normr = lsmr(sN11_R, -gc_r, damp=0.0, atol=1e-06, btol=1e-06, conlim=100000000.0, maxiter=None, show=False,  x0=d_mr1)[:4]
        d_mr1 = d_mr.copy()
        d_mr = d_mr.reshape(d_mr.shape[0],1)
        mu_LS = 1.0
        iarm = 1
        while 1:
            mr1 = mr + mu_LS*d_mr

            mr1[mr1 > np.log(10000)] = np.log(10000)
            mr1[mr1 < np.log(0.0001)] = np.log(0.0001)

            dr = forward(fobert1, mr1, size)
            dr = dr.reshape(dr.shape[0],1)

            dataerror_ert = rhos1 - dr
            fdert = (Wd.dot(dataerror_ert)).T.dot(Wd.dot(dataerror_ert))
            fmert = (L_mr*Wm.dot(mr1)).T.dot(L_mr*Wm.dot(mr1))
            ftert = (Wt.dot(mr1)).T.dot(Wt.dot(mr1))

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


        delta_mr = (mr-mr_R)


    ttt = np.reshape(mr,(-1,size),order='F')


    mesh2 = pg.load('inv_mulbnd1.bms')
    import meshop
    if nnn ==0:

        model2 = meshop.linear_interpolation(mesh, np.array(ttt[:,0]), mesh2)
        mesh2['res'] = np.exp(model2)
        mesh2.exportVTK(name[nnn][:-4])

    elif nnn == len(name) - 1:

        model2 = meshop.linear_interpolation(mesh, np.array(ttt[:,2]), mesh2)
        mesh2['res'] = np.exp(model2)
        mesh2.exportVTK(name[nnn+2][:-4])


    model2 = meshop.linear_interpolation(mesh, np.array(ttt[:,1]), mesh2)
    mesh2['res'] = np.exp(model2)
    mesh2.exportVTK(name[1+nnn][:-4])

    np.save(f[:-4]+str(nnn),dr)
    del mesh2
    gc.collect()

