"""
PHYS-512 PS4
Q2
Chuyang Li
"""
import numpy as np
import camb
from matplotlib import pyplot as plt
import time

def get_spectrum(pars,lmax=3000):
    #print('pars are ',pars)
    H0=pars[0]
    ombh2=pars[1]
    omch2=pars[2]
    tau=pars[3]
    As=pars[4]
    ns=pars[5]
    pars=camb.CAMBparams()
    pars.set_cosmology(H0=H0,ombh2=ombh2,omch2=omch2,mnu=0.06,omk=0,tau=tau)
    pars.InitPower.set_params(As=As,ns=ns,r=0)
    pars.set_for_lmax(lmax,lens_potential_accuracy=0)
    results=camb.get_results(pars)
    powers=results.get_cmb_power_spectra(pars,CMB_unit='muK')
    cmb=powers['total']
    tt=cmb[:,0]    #you could return the full power spectrum here if you wanted to do say EE
    return tt[2:]

#two-point derivative from problem set 1
def get_deriv(func, pars, index):
    #let dx = 0.01*x
    pars = pars
    p1 = pars.copy()
    p2 = pars.copy()
    p1[index] = 1.01*p1[index]
    p2[index] = 0.99*p2[index]
    dx = 0.01*pars[index]
    return (func(p1)-func(p2))/(2*dx) # (func(x+dx) - func(x-dx))/(2*dx)

def update_lamda(lamda,success):
    if success:
        lamda=lamda/1.5 #lamda divide by bigger than 1
        if lamda<0.5:
            lamda=0 #back to Newton's
    else:
        if lamda==0: 
            lamda=1
        else:
            lamda=lamda*1.5**2 #if Newton's fails, shrink step size by at least a factor of 2
    return lamda

def get_matrices(m,fun,y,npts,Ninv=None):
    model = fun(m)[:npts]
    derivs = np.zeros((npts, len(m)))
    for i in range(6):
         derivs[:,i] = get_deriv(fun, m, i)[:npts]
    r=y-model
    if Ninv is None:
        lhs=derivs.T@derivs
        rhs=derivs.T@r
        chisq=np.sum(r**2)
    else:
        lhs=derivs.T@Ninv@derivs
        rhs=derivs.T@(Ninv@r)
        chisq=r.T@Ninv@r
    return chisq,lhs,rhs

def linv(mat,lamda):
    mat=mat+lamda*np.diag(np.diag(mat))
    return np.linalg.inv(mat)

def fit_lm_clean(m,fun,y,npts,Ninv=None,niter=10,chitol=0.01):
#levenberg-marquardt fitter that doesn't wastefully call extra
#function evaluations, plus supports noise
    lamda=0
    chisq,lhs,rhs=get_matrices(m,fun,y,npts,Ninv)
    for i in range(niter):
        lhs_inv=linv(lhs,lamda)
        dm=lhs_inv@rhs
        chisq_new,lhs_new,rhs_new=get_matrices(m+dm,fun,y,npts,Ninv)
        if chisq_new<chisq:  
            #accept the step
            #check if we think we are converged - for this, check if 
            #lamda is zero (i.e. the steps are sensible), and the change in 
            #chi^2 is very small - that means we must be very close to the
            #current minimum
            if lamda==0:
                if (np.abs(chisq-chisq_new)<chitol):
                    print(np.abs(chisq-chisq_new))
                    print('Converged after ',i,' iterations of LM')
                    return m+dm, lhs
            chisq=chisq_new
            lhs=lhs_new
            rhs=rhs_new
            m=m+dm
            lamda=update_lamda(lamda,True)
            
        else:
            lamda=update_lamda(lamda,False)
        print('on iteration ',i,' chisq is ',chisq,' with step ',dm,' and lamda ',lamda)
    return m, lhs

planck=np.loadtxt('COM_PowerSpect_CMB-TT-full_R3.01.txt',skiprows=1)
spec=planck[:,1]
pars_guess = np.asarray([69,0.022,0.12,0.06,2.1e-9,0.95]) #starting guess, from Q1
errs = 0.5*(planck[:,2]+planck[:,3]) 
N = np.diag(errs**2) #noise matrix
N_inv = np.linalg.inv(N)

start_time = time.time()
best_pars, curv_matrix = fit_lm_clean(m=pars_guess, fun=get_spectrum, y=spec, npts=len(spec), Ninv=N_inv, niter=10, chitol=0.01)
print("it took %s seconds" %(time.time()-start_time))

cov_matrix = np.linalg.inv(curv_matrix)
errors = np.sqrt(np.diag(cov_matrix))

file1 = open("planck_fit_params.txt", "w")
file1.write("parameters:\n")
np.savetxt(file1, best_pars)
file1.write("errors:\n")
np.savetxt(file1, errors)
file1.close()
file2 = open("cov_matrix.txt", "w")
np.savetxt(file2, cov_matrix)
file2.close()











