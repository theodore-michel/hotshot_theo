#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 16 13:58:35 2020

@author: andrea
"""

import math
import numpy as np
import matplotlib.pyplot as plt
#from scipy.integrate import cumtrapz



def heter(N,DX):
# Adapted by AL from Jean-Paul Ampuero heter_stress routine
    # This function returns a time series of Gaussian distributed noise 
    # with colored amplitude spectrum normalized to unit variance

    L1 = np.Inf
    L2 = 0.0
    H = 0.5
    nr = 1
#    autocorr = 0
#    unify = 0
# wavenumbers
    dk = 2*np.pi/(N*DX) ;
    # Fix next line???
    k = np.zeros((N,1))
    k[0:np.int(np.floor(N/2))+1,0] = np.arange(0,N/2 + 1)
    k[np.int(np.floor(N/2))+1:,0] = np.arange(-N/2+1, 0)
    k *= dk
    
# von Karman amplitude spectrum 
    halfP = (H + 0.5)/2;
    K1 = 1/L1;
    k[0] = 0.00001 # To fix runtime warning in pow. It will be overwritten later
    fw = (K1**2.0 + k**2.0)**(-halfP); 
    if (K1==0):
        fw[0]=0
        k[0] = 0.0
        
# Gaussian filter
    if (L2 > 0):
        fw = fw * np.exp(-0.5 * (L2 * k)**2)  
        
# normalize to std=1
    fw = fw / np.std( np.real(np.fft.ifft(fw[:,0])) )


#-- Compute stochastic stress fields
# flat spectrum with random phase
    theta = np.random.uniform(0,1, size=(np.int(np.floor(N/2))-1, nr));
    ph = np.exp(1j*2.0*np.pi*theta);
    fs = np.zeros((N,1), dtype=complex)
    fs[1:len(ph)+1] = ph
    fs[len(ph)+2:] = np.conj(ph[::-1])
    
 # Note: zero amplitude at wavenumber=0 to produce the normalization mean=0
 # Note: spectral amplitude at Nyquist wavenumber must be real,
 #       here it is set to zero (but could instead be = +/- 1 with random sign)

# Combine random phase and von Karman/Gaussian spectrum
    fs = fs * fw
    stress = np.real(np.fft.ifft(fs[:,0]));
    
    return stress

    
    
    

def compute_STF(Mw, noise=False):
    
    
    # This function returns a source time function (STF) given Mw using the
    # functional form described in Meier, Ampuero and Heaton 2017
    
    # Optionally, specifying noise = True will add brownian noise to the STF
    
#    print('Calculating STF for Mw: ' + str(Mw))
    M0 = math.pow(10,1.5*Mw+9.1)
    max_time = 350.0
    dt = 1.0
    time = np.arange(0.0,max_time,dt)
#    dt = time[2]-time[1]
    mu = 1.25*10**18
#    mu = 1.0

    lambdar=10.0**(7.24-0.41*np.log10(M0))
    
#    Uncomment the following to include uncertainties on lambda
#    lambdar=10.0**(7.24-0.41*np.log10(M0)+ np.sqrt(0.25)*np.random.randn(1))

    ff = mu * time * np.exp(-0.5*(lambdar*time)**2)
#    sta = 0.01
#    nn = cumtrapz(sta * np.random.randn(len(time)), time, initial=0.0)
    nn2 = heter(len(time),dt)
    
#    plt.hist(nn)
#    print ('Variance of integrated Gauss: ' +str(np.var(nn)/sta**2))
    if noise:
        noise_term = 1.0 + (0.38 * (nn2 / np.std(nn2) ))
        # noise_term = 1.0 + (0.38 * (nn2))
        
        ff1 = ff * noise_term
        # With noise
        STF = ff1/(np.trapz(ff1,time))
#        plt.plot(time, STF)

    else:
        # Noise free source time function
        STF = ff/(np.trapz(ff,time))

  
    
    return STF


  







