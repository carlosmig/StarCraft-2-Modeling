# -*- coding: utf-8 -*-
"""

Hopf whole-brain model adapted from [1] and [2]. 
Used to simulate fMRI BOLD-like signals.

[1] Deco, G., Cruzat, J., Cabral, J., Tagliazucchi, E., Laufs, H., Logothetis, N. K., 
& Kringelbach, M. L. (2019). Awakening: Predicting external stimulation to force 
transitions between different brain states. Proceedings of the National Academy 
of Sciences, 116(36), 18088-18097.

[2] Escrichs, A., Perl, Y. S., Uribe, C., Camara, E., Türker, B., 
Pyatigorskaya, N., ... & Deco, G. (2022). Unifying turbulent dynamics 
framework distinguishes different brain states. Communications Biology,
 5(1), 638.

@author: Carlos Coronel

"""

import numpy as np
import networkx as nx
from numba import jit,float64
from numba.core.errors import NumbaPerformanceWarning
import warnings
warnings.simplefilter('ignore', category=NumbaPerformanceWarning)

#Simulation parameters
dt = 1E-1 #Integration step (seconds)
teq = 60 #Equilibrium time (seconds)
tmax = 600 #Signals' length (seconds)
downsamp = 1 #This reduces the number of points of the signals by a factor X

#Network parameters
nnodes = 90 #number of nodes
ones_vector = np.ones(nnodes).reshape((1,nnodes)) #trick for speed up simulations with Numba


#Model parameters
a = 0 * np.ones((nnodes,1)) #Bifurcation parameter (vector)
F0 = 0 * np.ones((nnodes,1)) #External stimulation (vector)
w = 0.05 * 2 * np.pi #Oscillatory frequency of each node (vector). The nodes
#Note: frequencies could be different between brain regions and also obtained using the real data
beta = 0.1 #noise scaling factor (background noise)
sigma = 0 #noise scaling factor (external stimulation noise)

#Structural connectivity matrix (toy matrix)
C = nx.to_numpy_array(nx.watts_strogatz_graph(nnodes,8,0.075,0))
norm = 1 #no normalization
G = 0 / norm #Global coupling
seed = 0 #Random seed

@jit(float64[:,:](float64[:,:],float64[:,:],float64,float64[:,:],float64,float64[:,:],float64[:,:]),nopython=True)
#Hopf multi-column model
def Hopf_model(x,y,t,C,G,a,F0):
    
    deltaX = (x @ ones_vector).T - (x @ ones_vector)
    deltaY = (y @ ones_vector).T - (y @ ones_vector)
    
    IsynX = G * C * deltaX @ ones_vector.T
    IsynY = G * C * deltaY @ ones_vector.T
    
    x_dot = (a - x**2 - y**2) * x - w * y + IsynX + F0 * np.cos(t * w)
    y_dot = (a - x**2 - y**2) * y + w * x + IsynY + F0 * np.sin(t * w)
    
    return(np.hstack((x_dot,y_dot)))


@jit(float64[:,:](float64, float64, float64[:,:]),nopython=True)
#Noise function
def noise(beta, sigma, F0):
    
    x_dot = np.random.normal(0, 1, (nnodes,1)) * (beta + sigma * F0) 
    y_dot = np.random.normal(0, 1, (nnodes,1)) * (beta + sigma * F0)
  
    return(np.hstack((x_dot,y_dot)))


@jit(float64(float64),nopython=True)
#This function is just for setting the random seed
def set_seed(seed):
    np.random.seed(seed)
    return(seed)


def update():
    Hopf_model.recompile()
    noise.recompile()
    set_seed.recompile()


def Sim(verbose = True):
    """
    Run a network simulation with the current parameter values
    using Euler-Maruyama.
    
    Note that the time unit in this model is seconds.

    Parameters
    ----------
    verbose : Boolean, optional
        If True, some intermediate messages are shown.
        The default is False.

    Raises
    ------
    ValueError
        An error raises if the dimensions of C and the number of nodes
        do not match.

    Returns
    -------
    x : ndarray
        Time trajectory for the x variable of each node.
    y : ndarray
        Time trajectory for the y variable of each node.    
    time_vector : numpy array (vector)
        Values of time.
        
    """
    global C, nnodes, seed, Neq, Nmax, teq, tmax, tsim
         
    if C.shape[0]!=C.shape[1] or C.shape[0]!=nnodes:
        raise ValueError("check C dimensions (",C.shape,") and number of nodes (",nnodes,")")
    
    if C.dtype is not np.dtype('float64'):
        try:
            C=C.astype(np.float64)
        except:
            raise TypeError("C must be of numeric type, preferred float")    
    
    set_seed(seed) #Set the random seed for "numbificated" functions  
    np.random.seed(seed)  #Set the random seed for anything else  
   
    Neq = int(teq / dt / downsamp) #Number of points to discard
    Nmax = int(tmax / dt / downsamp) #Number of points of the signals
    tsim = teq + tmax #total simulation time
    Nsim = int(tsim / dt) #total simulation points without downsampling
    
    #Initial conditions
    ic = np.ones((nnodes,2)) * np.random.uniform(0.01,1,((nnodes,2)))
    results = np.zeros((Nmax + Neq,nnodes,2))
    results[0,:,:] = np.copy(ic)
    results_temp = np.copy(ic) #Temporal vector to update y values 
    
    
    #Time vector
    time_vector = np.linspace(0, tmax, Nmax)

    if verbose == True:
        for i in range(1,Nsim):
            results_temp += Hopf_model(results_temp[:,[0]],
                                       results_temp[:,[1]],i*dt,C,G,a,F0) * dt + np.sqrt(dt) * noise(beta,sigma,F0)
            #This line is for store values each 'downsamp' points
            if (i % downsamp) == 0:
                results[i//downsamp,:,:] = results_temp
            if (i % (10 / dt)) == 0:
                print('Elapsed time: %i seconds'%(i * dt)) #this is for impatient people
    else:
        for i in range(1,Nsim):
            results_temp += Hopf_model(results_temp[:,[0]],
                                       results_temp[:,[1]],i*dt,C,G,a,F0) * dt + np.sqrt(dt) * noise(beta,sigma,F0)
            #This line is for store values each 'downsamp' points
            if (i % downsamp) == 0:
                results[i//downsamp,:,:] = results_temp
    
    results = results[Neq:,:,:]
    
    return(results, time_vector)







