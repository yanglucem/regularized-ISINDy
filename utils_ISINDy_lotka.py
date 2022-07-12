# -*- coding: utf-8 -*-
"""
Created on Fri Dec 17 10:01:08 2021

@author: yang
"""

import numpy as np
import tensorflow as tf
from scipy.integrate import solve_ivp
import sys
import win32api



# =============================================================================
# Define the ODE of grey verhulst
# =============================================================================
def verhulst(t,y,p):
    dy1 = p[0]*y[0] + p[1]*y[0]*y[1] 
    dy2 = y[0]  
    dy = np.array([dy1,dy2])
    
    return dy


# =============================================================================
# Define the ODE of grey verhulst
# =============================================================================
def grey_lotka(t,y,p):
    dy1 = p[0] * y[0] + p[1] * (y[0] * y[3] + y[1] * y[2])
    dy2 = p[2] * y[1] + p[3] * (y[0] * y[3] + y[1] * y[2])
    dy3 = y[0]
    dy4 = y[1]
    
    # notice y[0] is x1, y[1] is x2, y[2] is y1, y[3] is y2
    dx= [dy1,dy2,dy3,dy4]
    return dx


# =============================================================================
# Define the ODE of 2-nd lib
# ============================================================================
def lotka(t,y,p):
    dy1 = p[0,0]*y[0] + p[1,0]*y[1] + p[2,0]*y[0]*y[3] + p[3,0]*(y[0]*y[3] + y[1]*y[2]) + p[4,0]*y[1]*y[3]
    dy2 = p[0,1]*y[0] + p[1,1]*y[1] + p[2,1]*y[0]*y[3] + p[3,1]*(y[0]*y[3] + y[1]*y[2]) + p[4,1]*y[1]*y[3]
    dy3 = y[0]
    dy4 = y[1]
    
    dx= [dy1,dy2,dy3,dy4]
    
    return dx


# =============================================================================
# Define the polinomial library
# =============================================================================
def lib(y,liborder,tobs):
        
    
    # get the dimensional of y
    n,d = y.shape
    h = np.diff(tobs)
    
    # approxitmate the integral by Trapezoidal rule
    z=np.zeros((n,d))
    for ii in range(d):
        z[1:n+1,ii] = ((np.cumsum(h * y[1:n,ii]) + np.cumsum(h * y[0:n-1,ii])) / 2)
    
    # Lib order 0
    # Theta=np.ones((n,1),dtype=float)
    Theta=z[:,[0]]
    
    if liborder>=1:
        for i in range(1,d):
            Theta = np.concatenate((Theta,z[:,[i]]),axis=1)
                
    if liborder>=2:
        for i in range(d):
            for j in range(i,d):
                if i == j:
                    new = .5*z[:,[i]]*z[:,[j]]
                    Theta = np.concatenate((Theta,new),axis=1)
                else:
                    new = z[:,[i]]*z[:,[j]]
                    Theta = np.concatenate((Theta,new),axis=1)
                    
                
    if liborder>=3:
        for i in range(d):
            for j in range(i,d):
                for k in range(j,d):
                    new = z[:,[i]]*z[:,[j]]*z[:,[k]]
                    Theta = np.concatenate((Theta,new),axis=1)
                    
    if liborder>=4:
        for i in range(d):
            for j in range(i,d):
                for k in range(j,d):
                    for l in range(k,d):
                        new = z[:,[i]]*z[:,[j]]*z[:,[k]]*z[:,[l]]
                        Theta = np.concatenate((Theta,new),axis=1)
                        
    if liborder>=5:
        for i in range(d):
            for j in range(i,d):
                for k in range(j,d):
                    for l in range(k,d):
                        for m in range(l,d):
                            new = z[:,[i]]*z[:,[j]]*z[:,[k]]*z[:,[l]]*z[:,[m]]
                            Theta = np.concatenate((Theta,new),axis=1)
                            
    return Theta    


# =============================================================================
# calculate the discretization of the integral term, the length is datalen
# =============================================================================
@tf.function
def theta_element(y,h):
    
    n,d = y.shape
    z = tf.constant(h/2)*tf.cumsum(y[1:n],axis=0) + tf.constant(h/2)*tf.cumsum(y[0:n-1],axis=0)
    xint = tf.concat([tf.zeros((1,d)),z],axis=0)
       
    return xint


# =============================================================================
# define the GPU version of lib
# =============================================================================
@tf.function
def libGPU(z):
    
    # # 1-dimensional, 3nd order  
    # Theta = tf.concat([z,.5*z**2,1/3*z**3],axis=1)
    
    # 2-dimensional, 3nd order
    z1=tf.gather(z,[0],axis=1)      # indices
    z2=tf.gather(z,[1],axis=1)
    
    Theta=tf.concat([z1,z2,.5*z1**2,z1*z2,.5*z2**2,z1**3,
                     1/3*(z1**2)*z2,1/3*z1*(z2**2),1/3*z2**3],axis=1)
    
    return Theta


@tf.function
def RK4_y(x,h):
    
    n,d = x.shape
    f = tf.slice(x,[0,0],[n-1,d])
    k1 = f*tf.constant(1/6*h)
    k2 = tf.math.add(f,tf.constant(h*0.5)*k1)*tf.constant(1/3*h)
    k3 = tf.math.add(f,tf.constant(h*0.5)*k2)*tf.constant(1/3*h)
    k4 = tf.math.add(f,tf.constant(h)*k3)*tf.constant(1/6*h)
    
    y = tf.math.add_n([f,k1,k2,k3,k4])
    
    return tf.concat([tf.zeros([1,d]),y],axis=0)




              
# =============================================================================
# Define the basic integral-SINDy regression, using y
# =============================================================================
def ISINDy(y,Theta,h,lam,dims):
    n,m=Theta.shape
        
    Theta = np.concatenate((Theta,np.ones((n,1))),axis=1)
    
    # initial guess: least-squares
    Xi = np.dot(np.dot(np.linalg.pinv(np.dot(Theta.T,Theta)),Theta.T),y)
    
    for i in range(10):
        smallinds = (np.abs(Xi)<lam)
        Xi[smallinds] = 0
        for ind in range(dims):
            biginds = ~smallinds[:,ind]
            # regress gynamics onto remaining terms to find sparse Xi
            Xi[biginds,ind] = np.dot(np.dot(np.linalg.pinv(np.dot(Theta[:,biginds].T,Theta[:,biginds])),
                    Theta[:,biginds].T),y)[:,ind]
            
    pars = Xi[0:len(Xi)-1,:]
    ini = Xi[len(Xi)-1:len(Xi)]

    return pars,ini




# =============================================================================
# Define the weight of decay 
# =============================================================================
def decayfactor(ro,dims,q):
    if q == 0:
        weigths = 1
    elif q<0:
        raise Exception("The prediction step must be equals or greater than zero")
    else:
        weigths = []
        for i in range(q):
            for j in range(dims):
                weigths = np.append(weigths,ro**(i))
                
    return weigths


# =============================================================================
# Slice the data in to backward and forward matrix, need a GPU version
# =============================================================================
def slicedata(y,q,nlen):
    if q == 0:
        yfward = y
        ybward = y
    elif q<0:
        raise Exception("The prediction step must be equals or greater than zero")
    else:
        yfward = []
        ybward = []
        for i in range(1,q+1):
            if i == 1:
                yfward = y[q+i:nlen-q+i,:]
                ybward = y[q-i:nlen-q-i,:]
            else:
                yfward = np.append(yfward,y[q+i:nlen-q+i,:],axis=1)
                ybward = np.append(ybward,y[q-i:nlen-q-i,:],axis=1)
                
    return ybward, yfward
    

# =============================================================================
# Slice the noise data in to backward and forward matrix
# =============================================================================
@tf.function
def slicenoise(noisevar,q,nlen,dims):
    if q==0:
        noisefward = noisevar
        noisebward = noisevar
    elif q<0:
        raise Exception("The prediction step must be equals or greater than zero")
    else:
        noisefward = tf.slice(noisevar,[q+1,0],[nlen-2*q,dims])
        noisebward = tf.slice(noisevar,[q-1,0],[nlen-2*q,dims])

        # slice(input_,begin,size,name=None)
        
        for i in range(1,q):
            noisefward = tf.concat([noisefward,tf.slice(noisevar,[q+1+i,0],
                                                        [nlen-2*q,dims])],axis=1)
            noisebward = tf.concat([noisebward,tf.slice(noisevar,[q-1-i,0],
                                                        [nlen-2*q,dims])],axis=1)
       
    return noisebward, noisefward



# =============================================================================
# Define the F^q function and slice it into forward and backeward part
# =============================================================================
@tf.function
def sliceF(xmid,xint,libGPU,q,nlen,dims,Xi):
        
    Fbward = tf.zeros_like(xmid)
    Ffward = tf.zeros_like(xmid)
    
    for i in range(q):
        if i==0:
            Fbward = F_backward(xmid,xint,libGPU,Xi,q,i,nlen,dims)
            Ffward = F_foreward(xmid,xint,libGPU,Xi,q,i,nlen,dims)  
        else:
            Fbward = tf.concat([Fbward,
                                F_backward(xmid,xint,libGPU,Xi,q,i,nlen,dims)],axis=1)
            Ffward = tf.concat([Ffward,
                                F_foreward(xmid,xint,libGPU,Xi,q,i,nlen,dims)],axis=1)
        
    
    return Fbward, Ffward


@tf.function
def F_backward(xmid,xint,libGPU,Xi,q,i,nlen,dims):
    
    K1 = -tf.linalg.matmul(libGPU(tf.slice(xint,[q,0],[nlen-q*2,dims])),Xi)
    K2 = tf.linalg.matmul(libGPU(tf.slice(xint,[q-i-1,0],[nlen-q*2,dims])),Xi)
        
    
    return tf.math.add_n([xmid,K1,K2])
    


@tf.function
def F_foreward(xmid,xint,libGPU,Xi,q,i,nlen,dims):
    
    K1 = -tf.linalg.matmul(libGPU(tf.slice(xint,[q,0],[nlen-q*2,dims])),Xi)
    K2 = tf.linalg.matmul(libGPU(tf.slice(xint,[q+i+1,0],[nlen-q*2,dims])),Xi)
    
    return tf.math.add_n([xmid,K1,K2])



# =============================================================================
# Define the iterated loss function
# =============================================================================
def train_nss_ISINDy(yobs,yfward,ybward,noisevar,Xi,eta,weights,h,q,nlen,dims,
                     optimizer,n_train):     
    
    ''' calculate the cost and update the gradient'''
    for i in range(n_train):
        
        
        # training the loss function
        lossfun = onestep_ro_ISINDy(yobs,yfward,ybward,noisevar,Xi,eta,weights,
                                    h,q,nlen,dims,optimizer,libGPU,n_train)
        
        if i%500==0:
            tf.print(lossfun)
             
    return noisevar.numpy(),lossfun.numpy()
    

# =============================================================================
# Define the onestep loss function
# =============================================================================
@tf.function
def onestep_ro_ISINDy(yobs,yfward,ybward,noisevar,Xi,eta,weights,h,q,nlen,dims,
                      optimizer,libGPU,n_train):

    with tf.GradientTape() as g:
        
        # get the initial condition guess, we estimate by ISINDy
        x = tf.constant(yobs,dtype=tf.dtypes.float32)
        xhat = tf.math.subtract(x,noisevar)
        xmid = tf.slice(xhat,[q,0],[nlen-2*q,dims])
        xint = theta_element(xhat,h)
        
        
        # simulate the system forward and bacward
        Fbward,Ffward = sliceF(xmid,xint,libGPU,q,nlen,dims,Xi)
        
        
        # get the forward and backward noise 
        nbward,nfward = slicenoise(noisevar,q,nlen,dims)
        
        # calculate the F error
        ybpre = nbward + Fbward
        yfpre = nfward + Ffward
        Fback_loss = tf.reduce_mean(tf.math.multiply(
                            tf.math.squared_difference(ybward,ybpre),weights))
        Ffore_loss = tf.reduce_mean(tf.math.multiply(
                            tf.math.squared_difference(yfward,yfpre),weights))
        loss_noise = tf.math.add(Fback_loss,Ffore_loss)
        
        # calculate the state variable error  
        mat_theta = tf.linalg.matmul(libGPU(xint),Xi)
        add_theta = tf.math.add(mat_theta,eta)
        loss_state = tf.reduce_mean(tf.math.squared_difference(xhat,add_theta))
        
    
        
        # note: in this loop, Xi is changed
        
        
        lossfun = tf.math.add(loss_noise,loss_state)
        
    # calculate the gradient with respect to variables
    gard = g.gradient(lossfun,[Xi,noisevar])
    optimizer.apply_gradients(zip(gard,[Xi,noisevar]))
    
    return lossfun


    




    