# -*- coding: utf-8 -*-
"""
Created on Tue Mar 29 20:41:41 2022

@author: yang
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Feb 28 11:01:31 2022

@author: yang

Code for paper 'Robust data-driven discovery of nonlinear integro-differential equation-based grey system models'
"""

import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from fitter import Fitter
from utils_ISINDy_lotka import *

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# =============================================================================
# define the parameters of the simulation
# =============================================================================

# prepare for tensorflow
datatype = tf.dtypes.float32

# define the parameters and initial value
pars = np.array([2.,-.5,-4,.5] )
x0 = np.array([1,3])
y0 = np.array([0,0])

# define the bool version of the parameter
par_tru = np.array([[ True, False],
                   [False,  True],
                   [False, False],
                   [ True,  True],
                   [False, False],
                   [False, False],
                   [False, False],
                   [False, False],
                   [False, False]])


# define the time points, take n = 501 as an example
T = 5   
h= 0.01 
tobs = np.linspace(0,T,int(T/h)+1)

# define the noise level, take 5% as an example
noisepercentage = 5 

# =============================================================================
# now simulate the real system
# =============================================================================

solu = solve_ivp(grey_lotka,[0,T],np.append(x0,y0),args=(pars,),
                     dense_output=True,method='LSODA')
solu_value = solu.sol(tobs).T
xobs = solu_value[:,0:2]
nlen,dims = xobs.shape 

# define the q step
q = 1
ro = 0.9
weights = tf.constant(decayfactor(ro,dims,q),dtype=datatype)
    
liborder = 3
lam = 0.2


# =============================================================================
# generate noise, Gussian
# =============================================================================  
np.random.seed(1)
noisemap = [np.std(xobs[:,i])*noisepercentage*.01 for i in range(dims)]
noise = np.hstack([noisemap[i]*np.random.randn(nlen,1) for i in range(dims)])
yobs = xobs + noise    


# get the forward and backward measurement data (output tensor, will be trained in the main loop)
ybward,yfward = slicedata(yobs,q,nlen)
ybward=tf.constant(ybward,dtype=datatype)
yfward=tf.constant(yfward,dtype=datatype)

       
        
# =============================================================================
# initial guess of the noise part
# =============================================================================

# hard start: assume the noise begin with 0 vector
noise0 = np.zeros((nlen,dims))
xhat = yobs - noise0

# get the initial guess of the noise, we make a random guess here.
# for better performance we could first smooth the data and guess the noise 
noisevar = tf.Variable(noise0,dtype=datatype)
    
# =============================================================================
# initial guess of parameters and initial value
# =============================================================================
Theta = lib(xhat,liborder,tobs)
Xi0,eta0 = ISINDy(xhat,Theta,h,lam,dims)
print (Xi0)
print(eta0)
Xi = tf.Variable(Xi0,dtype=datatype)
eta = tf.Variable(eta0,dtype=datatype)
    
    
# =============================================================================
# define the adam optimizer
# =============================================================================
tf.config.run_functions_eagerly(True)

optimizer = tf.keras.optimizers.Adam(learning_rate=0.001,epsilon=1e-09)


n_train0 = 1
loss = onestep_ro_ISINDy(xhat,yfward,ybward,noisevar,Xi,eta,weights,h,q,nlen,dims,
                      optimizer,libGPU,n_train0)
loss = loss.numpy()

# =============================================================================
# start the main loop
# =============================================================================
nloop = 1
n_train = 2000

while loss > 10**(-3):
    print("runing the loop",str(nloop))
    nloop = nloop + 1
    
    noise_id,loss = train_nss_ISINDy(yobs,yfward,ybward,noisevar,Xi,eta,weights,h,q,nlen,dims,
                     optimizer,n_train)
    print("Xi result obtained by Adam")
    print(Xi)
    print(eta)
    
    
    # note each iteration use yobs
    xhat = yobs - noise_id
    
    
    # Do ISINDy on the denoised data
    Theta_iter = lib(xhat,liborder,tobs)
    Xi_new,eta_new = ISINDy(xhat,Theta_iter,h,lam,dims)
    
    print("current Xi result constrain by ISINDy")
    print(Xi_new) 
    print(eta_new)
    
    
    Xi = tf.Variable(Xi_new,dtype=tf.dtypes.float32)
    eta  = tf.Variable(eta_new,dtype=tf.dtypes.float32)
  
  
Xi_act = abs(Xi.numpy())>0

if (Xi_act == par_tru).all():
    
    Xi_hat = Xi.numpy()
    pars_hat = Xi_hat.T.ravel()[np.flatnonzero(Xi_hat.T)]
    eta_hat = eta.numpy()
    x_traj = solve_ivp(grey_lotka,[0,T],np.append(eta_hat,y0),args=(pars_hat,),
                      dense_output=True,method='RK45')
    solu_value = x_traj.sol(tobs).T
    xtraj = solu_value[:,0:2]
    
    # calculate the perpformance
    noise_erro = np.linalg.norm(noise - noise_id,'fro')**2/nlen
    state_erro = np.linalg.norm(xobs - xtraj,'fro')**2/np.linalg.norm(xobs,'fro')**2
    pars_erro = np.linalg.norm(pars - pars_hat)**2/np.linalg.norm(pars)**2
    eta_erro = np.linalg.norm(x0 - eta)**2/np.linalg.norm(x0)**2
    
    