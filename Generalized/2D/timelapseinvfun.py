import pygimli as pg
from pygimli.physics import ert  # the module
import numpy as np
from scipy.linalg import block_diag
import invesfun as inv2
from scipy.sparse.linalg import lsqr
import pygimli.meshtools as mt
import gc

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

    obs1 = obs[0].reshape(-1,1)
    for i in range(size-1):
        obs1 = np.vstack((obs1,obs[i+1].reshape(-1,1)))

    return obs1, JJ

def forward(fobert1, mr, mesh, size):
    ttt = np.reshape(mr,(-1,size),order='F')
    obs = []

    for i in range(size):

        dr = inv2.ertforward2(fobert1[i], ttt[:,i], mesh)
        obs.append(dr)


    obs1 = obs[0].reshape(-1,1)
    for i in range(size-1):
        obs1 = np.vstack((obs1,obs[i+1].reshape(-1,1)))
    return obs1


def timelapsefun(nnn,ertfile,new_Data_arr,size,Lambda,alpha,decay_rate):


    

    ## load ERT data file, note in Bert format
    rhos = []
    dataert1 = []
    dataerr = []

    for i in range(size):
        f = ertfile[nnn+i]
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

    ## load mesh and get the orignal marker value
    mesh = pg.load('inv.bms')



    fobert1= []
    ## forward operator
    for i in range(size):
        fobert = ert.ERTModelling()
        fobert.setData(dataert1[i])
        fobert.setMesh(mesh)
        fobert1.append(fobert)



    # initial model, here I try to give borehore and background big resistivity
    rhomodeltemp = []
    for i in range(size):
        rhomodel = np.median(rhos[i])*np.ones((fobert.paraDomain.cellCount(),1))
        rhomodeltemp.append(rhomodel)


    xr1 = np.log(np.reshape(np.array(rhomodeltemp),(fobert.paraDomain.cellCount()*size,1)))
    
    del rhomodeltemp
    
    mr = xr1
    mr_R = mr
    mr_R = mr_R.reshape(mr_R.shape[0],1)




    from scipy.sparse import diags
    from scipy.sparse import coo_matrix#, block_diag
    
    #---------------------data weight------------------------------------#
    dataerr = np.array(dataerr)
    err_temp = dataerr[0]
    for i in range(size-1):
        err_temp = np.hstack((err_temp,dataerr[i+1]))

    Delta_rhoa_rhoa = err_temp
    Wdert = np.diag(1.0 / np.log(Delta_rhoa_rhoa + 1))
    Wd = Wdert.copy()
    del Wdert
    gc.collect()

    # ------------------------model weighting matrix--------------------- #
    rm = fobert.regionManager()
    Ctmp = pg.matrix.RSparseMapMatrix()


    rm.setConstraintType(1)
    rm.fillConstraints(Ctmp)
    Wm_r = pg.utils.sparseMatrix2coo(Ctmp)

    #Wm_r = Wm_r.todense()

    cw = rm.constraintWeights().array()

    ### just adding structure information here, you can ignore it and delete in your own code

    Wm_r = diags(cw).dot(Wm_r)

    Wm = Wm_r.todense()
    for i in range(size-1):
        Wm = block_diag(Wm,Wm_r.todense())

    del Wm_r

    gc.collect()
    # ------------------------time space weighting matrix--------------------- #
    Wt = np.zeros((int(len(mr)*(size-1)/size),len(mr)))

    np.fill_diagonal(Wt[:,int(len(mr)*1/size):], -1) ## adjanct time model paramters
    np.fill_diagonal(Wt, 1)

    #NNoid = np.vstack((Noid.reshape(-1,1),Noid.reshape(-1,1),Noid.reshape(-1,1)))



    L_mr = Lambda
    alpha = alpha#1 #5
    mr = mr.reshape(mr.shape[0],1)
    chi2_ert = 1

    
    
    tdiff = np.diff(new_Data_arr[nnn:nnn+size]) # the time difference between ERT data

    w_temp = np.ones(int(len(mr)*1/size))*alpha*np.exp(decay_rate*tdiff[0].data[0]/60)
    for i in range(size-2):
        w_temp = np.hstack((w_temp,np.ones(int(len(mr)*1/size))*alpha*np.exp(decay_rate*tdiff[i+1].data[0]/60)))

    beta = 1 # in case to help inversion converge to a satisfied vaule, otherwise, keep 1
    coeff = w_temp

    Wt = np.diagflat(coeff).dot(Wt)
    # WWd = Wd.T.dot(Wd)
    #
    # WWm = (L_mr*Wm).T.dot(L_mr*Wm)
    # WWt = Wt.T.dot(Wt)
    
    del coeff
    gc.collect()
    for nn in range(20):
        print('-------------------ERT Iteration:'+str(nn)+'---------------------------')

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
        print('dPhi:'+str(dPhi))


        if chi2_ert < 1 or dPhi < 0.01:
            break

        gc_r = beta*Jr.T.dot(Wd.T.dot(Wd)).dot(dr-rhos1) + (L_mr*Wm).T.dot(L_mr*Wm).dot(mr) + Wt.T.dot(Wt).dot(mr)
        N11_R = coo_matrix(beta*Jr.T.dot(Wd.T.dot(Wd)).dot(Jr)) + coo_matrix((L_mr*Wm).T.dot(L_mr*Wm)) + coo_matrix(Wt.T.dot(Wt))

        del Jr
        gc.collect()
        gc_r = np.array(gc_r)
        gc_r = gc_r.reshape((-1,))
        #sN11_R = N11_R
        #gc_r1 = gc_r # + alpha*Wt.T.dot(Wt).dot(mr)
        
         
        d_mr, istop, itn, normr = lsqr(N11_R, -gc_r, damp=0.0, atol=1e-08, btol=1e-08, conlim=100000000.0,
                                       iter_lim=150, show=False, calc_var=False, x0=None)[:4]
        
        del N11_R, istop, itn, normr

        d_mr = d_mr.reshape(d_mr.shape[0],1)
        mu_LS = 1.0
        iarm = 1
        while 1:
            mr1 = mr + mu_LS*d_mr

            mr1[mr1 > np.log(10000)] = np.log(10000)
            mr1[mr1 < np.log(0.0001)] = np.log(0.0001)

            dr = forward(fobert1, mr1, mesh,size)
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

        mr = mr1
        mr[mr > np.log(10000)] = np.log(10000)
        mr[mr < np.log(0.0001)] = np.log(0.0001)

        # if L_mr > 200:
        #     L_mr = L_mr*0.5
        #
        # delta_mr = (mr-mr_R)



    ################## do the interpolation and save the results################
    ttt = np.reshape(mr,(-1,size),order='F')
    mesh2 = pg.load('inv.bms')
    name = ertfile
    import meshop
    if nnn == 0:
        for i in range(int(size/2)):
            model2 = meshop.linear_interpolation(fobert.paraDomain, np.array(ttt[:,i]), mesh2)
            mesh2['res'] = np.exp(model2)
            mesh2.exportVTK(name[nnn][:-4])

    elif nnn == len(name) - int(size/2):
        for i in range(int(size/2)):
            model2 = meshop.linear_interpolation(fobert.paraDomain, np.array(ttt[:,size - 1 - i]), mesh2)
            mesh2['res'] = np.exp(model2)
            mesh2.exportVTK(name[nnn+2][:-4])


    model2 = meshop.linear_interpolation(fobert.paraDomain, np.array(ttt[:,int(size/2)]), mesh2)
    mesh2['res'] = np.exp(model2)
    mesh2.exportVTK(name[1+nnn][:-4])

    np.save(f[:-4]+str(nnn),dr)
    del mesh2
    gc.collect()
