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
from scipy import signal
import time
import Hopf_model_multi as HM
import matplotlib.pyplot as plt
import importlib
importlib.reload(HM)


#Simulation parameters
HM.dt = 1E-1 #Integration step (seconds)
HM.teq = 10 #Equilibrium time (seconds)
HM.tmax = 900 #Signals' length (seconds)
HM.downsamp = 1 #This reduces the number of points of the signals by a factor X
HM.seed = 0 #random seed

#Network parameters
HM.C = np.loadtxt('structural_Deco_AAL.txt')
HM.nnodes = len(HM.C)
HM.norm = 1 #global normalization
HM.G = 0 #Global coupling
HM.ones_vector = np.ones(HM.nnodes).reshape((1,HM.nnodes)) #trick for speed up simulations with Numba


#Stimulation parameters
stim_rois = [10,11,12,79,78,77] #indexes of the ROIs to be stimulated
HM.a = 0 * np.ones((HM.nnodes,1)) #Bifurcation parameter
HM.F0 = 0 * np.ones((HM.nnodes,1)) #External stimulation
pulse = 0 #Stimulation magnitude
HM.F0[stim_rois] += pulse #updating the pulse

#Model parameters
HM.w = 0.05 * 2 * np.pi #Oscillatory frequency
#Note: frequencies could be different between brain regions and also obtained using the real data
HM.beta = 0.1 #noise scaling factor (background noise)
sigma = 0 #noise scaling factor (external stimulation noise)

#Update parameters that aren't inputs of model's functions (e.g. oscillatory frequency)
HM.update()

#Simulation starts here
init = time.time()
y, time_vector = HM.Sim(verbose = False) #True showd time, for anxious people
BOLD_signals = y[:,:,0] #fMRI BOLD-like output of the model
end = time.time()
#Simulation ends here
print('Sim time: %.3f'%(end-init))

#Filtering BOLD signals
resolution = HM.dt * HM.downsamp
Fmin, Fmax = 0.01, 0.1 #Allowed frequencies

#Filter parameters
a0,b0 = signal.bessel(3,[2 * resolution * Fmin, 2 * resolution * Fmax],btype='bandpass')
BOLD_filt = signal.filtfilt(a0,b0,BOLD_signals,axis=0) #filtered signals

FC = np.corrcoef(BOLD_filt.T) #Functional Connectivity (FC) matrix

#%%

##some plots

plt.figure(1, figsize = (12,4))
plt.clf()
plt.subplot(1,2,1)
plt.plot(time_vector, BOLD_filt)
plt.xlabel('Time (sec)')
plt.ylabel('BOLD amplitude')
plt.title('Simulated fMRI BOLD-like signals')

plt.subplot(1,3,3)
plt.imshow(FC, vmin = 0, vmax = 1, cmap = 'jet')
plt.xlabel('ROIs')
plt.ylabel('ROIs')
plt.title('FC matrix')


