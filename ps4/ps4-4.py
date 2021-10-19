"""
PHYS-512 PS4
Q4
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

def spectrum_chisq(pars,y,npts,noise=None):
    pred=get_spectrum(pars)[:npts]
    if noise is None:
        return np.sum((y-pred)**2)
    else:
        return np.sum (((y-pred)/noise)**2)

def mcmc(pars,step_size,y,fun,npts,nstep=1000,noise=None):
    chi_cur=fun(pars,y,npts,noise)
    npar=len(pars)
    chain=np.zeros([nstep,npar+1])
    # chivec=np.zeros(nstep)
    count=0
    for i in range(nstep):
        trial_pars=pars+0.8*step_size@np.random.randn(npar)
        trial_pars[3] = 0.054 #fix tau = 0.054
        trial_chisq=fun(trial_pars,y,npts,noise)
        delta_chisq=trial_chisq-chi_cur
        accept_prob=np.exp(-0.5*delta_chisq) #decide if i take the step; if trial < current then it's a good step
        accept=False
        if np.random.rand(1)<accept_prob: 
            accept=True
            count = count + 1
        # accept=np.random.rand(1)<accept_prob #np.random.rand(1) is usually very small
        if accept:
            pars=trial_pars
            chi_cur=trial_chisq
        chain[i,1:]=pars
        # chivec[i]=chi_cur
        chain[i,0]=chi_cur
    accept_rate = count/nstep
    # return chain,chivec,accept_rate
    return chain, accept_rate

planck=np.loadtxt('COM_PowerSpect_CMB-TT-full_R3.01.txt',skiprows=1)
spec=planck[:,1]
errs=0.5*(planck[:,2]+planck[:,3]);
cov_matrix = np.loadtxt("cov_matrix.txt") #covariance matrix from #2
L = np.linalg.cholesky(cov_matrix)
pars_guess = [6.823463325770070753e+01, 2.236258621308569866e-02, 1.176838874120168799e-01,
              8.515373335540617206e-02, 2.218177568678960319e-09, 9.730110942363249249e-01] #parameters from #2
# trial_step = L@np.random.randn(len(pars_guess))

start_t = time.time()
chain, accept_percentage = mcmc(pars=pars_guess, step_size=L, y=spec, fun=spectrum_chisq, npts=len(spec), nstep=40, noise=errs)
print("it took %s seconds" %(time.time()-start_t))
file = open("planck_chain_tauprior.txt", "w")
np.savetxt(file, chain)
file.close()

