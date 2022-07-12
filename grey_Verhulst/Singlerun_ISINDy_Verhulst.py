  # -*- coding: utf-8 -*-
"""
Created on Thu Dec 16 18:34:53 2021

@author: yang
"""

import numpy as np
import tensorflow as tf
from scipy.integrate import solve_ivp
from utils_ISINDy_Verhulst import *
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# =============================================================================
# define the parameters of the simulation
# =============================================================================
# prepare for tensorflow
datatype = tf.dtypes.float32

# Check the GPU status
CheckGPU()

# define the parameters and initial value
pars = np.array([0.8, -0.2])
x0 = 0.6
y0 = 0
init = np.array([.6, 0])

# define the bool version of the parameter
par_tru = np.array([[True], [True], [False]])

# define the time points, take n = 101 as an example
T = 5
h = 0.05    
tobs = np.linspace(0,T,int(T/h)+1)

# define the noise level, take 10% as an example
noisepercentage = 10 


# =============================================================================
# now simulate the real system
# =============================================================================
solu = solve_ivp(verhulst, [0,T], init, args=(pars,), dense_output=True,
                method='RK45')
solu_value = solu.sol(tobs).T
xobs = solu_value[:,[0]]
nlen,dims = xobs.shape 


# define the q step
q = 1
ro = 0.9
weights = tf.constant(decayfactor(ro,dims,q),dtype=datatype)

# set the degree of library
liborder = 3
lam = 0.1


# =============================================================================
# generate noisy data, Gussian
# =============================================================================  
np.random.seed(1)
noisemap = [np.std(xobs)*noisepercentage*.01]
noise = noisemap*np.random.randn(nlen,1)
yobs = xobs + noise


# get the forward and backward measurement data (output tensor, will be trained in the main loop)
ybward,yfward = slicedata(yobs,q,nlen)
ybward=tf.constant(ybward,dtype=datatype)
yfward=tf.constant(yfward,dtype=datatype)


# =============================================================================
# initial guess of the noise part
# =============================================================================
# hard start
noise0 = np.zeros((yobs.shape[0],yobs.shape[1]))
xhat = yobs - noise0


# get the initial guess of the noise, we make a random guess here.
# for better performance we could first smooth the data and guess the noise 
noisevar = tf.Variable(noise0,dtype=datatype)


# =========================================================================
# initial guess of the parameter and initial condition
# =========================================================================
Theta = lib(xhat,liborder,tobs)
Xi0,eta0 = ISINDy(xhat,Theta,h,lam,dims)
print (Xi0)
print (eta0)
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
    
    noise_id,loss = train_nss_ISINDy(yobs,yfward,ybward,noisevar,Xi,eta,
                                     weights,h,q,nlen,dims,optimizer,n_train)
    
    # note each iteration use yobs
    xhat = yobs - noise_id
    
    # Do ISINDy on the denoised data
    Theta_iter = lib(xhat,liborder,tobs)
    Xi_new,eta_new =  ISINDy(xhat,Theta_iter,h,lam,dims)
    
    print("current Xi result constrain by ISINDy")
    print(Xi_new)  
    print(eta_new)
    
    Xi = tf.Variable(Xi_new,dtype=tf.dtypes.float32)
    eta  = tf.Variable(eta_new,dtype=tf.dtypes.float32)
   

# tell whether the identified structure is right
Xi_act = abs(Xi.numpy())>0

if (Xi_act == par_tru).all():
    
    Xi_hat = Xi.numpy()
    pars_hat = Xi_hat.T.ravel()[np.flatnonzero(Xi_hat.T)]
    eta_hat = eta.numpy()
    x_traj = solve_ivp(verhulst,[0,T],np.append(eta_hat,y0),args=(pars_hat,),
                      dense_output=True,method='RK45')
    solu_value = x_traj.sol(tobs).T
    xtraj = solu_value[:,0:1]
    
    # calculate the perpformance
    noise_erro = np.linalg.norm(noise - noise_id)**2/nlen
    state_erro = np.linalg.norm(xobs - xtraj)**2/np.linalg.norm(xobs)**2
    pars_erro = np.linalg.norm(pars - pars_hat)**2/np.linalg.norm(pars)**2
    eta_erro = np.linalg.norm(x0 - eta)**2/np.linalg.norm(x0)**2
     