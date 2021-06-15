#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  3 16:15:16 2020

@author: licciar
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import h5py
import scipy.stats as st
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pegslib_NEZ_one

from sklearn.neighbors import DistanceMetric
from math import radians


import torch
from Classes import HDF5Dataset, VAE
from scipy import signal
from matplotlib.gridspec import GridSpec






def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx


def plot_comparison(label, noisy, reconstructed, STAT, ptimes, eq_params, samp_ID, stat_ID):
    
    # INPUTS
    #
    # label [NEXAMP, n_chan, t_max, n_stat]: noise-free images
    # noisy [NEXAMP, n_chan, t_max, n_stat]: noise corrupted images
    # reconstructed [NEXAMP, n_chan, t_max, n_stat]: reconstructed images
    # samp ID: index of the sample to plot for the current batch
    # stat_ID: index of the sample for station
    
    # Rearrange the input
    
    label = label[samp_ID, 0, :,:]
    label = np.swapaxes(label,0,1)
    
    noisy = noisy[samp_ID, 0, :,:]
    noisy = np.swapaxes(noisy,0,1)
    
    reconstructed = reconstructed[samp_ID, 0, :,:]
    reconstructed = np.swapaxes(reconstructed,0,1)
    
    vmin = -0.1*1.0e-9
    vmax = 0.0*1.0e-9
    
    ########################
    fig = plt.figure(constrained_layout=False)
    gs = GridSpec(3, 2, figure=fig)
    ax1 = fig.add_subplot(gs[0,:])
    ax1.hlines(stat_ID,0,noisy.shape[1],'r')
    im = plt.imshow(noisy, vmin=vmin,vmax=vmax)
    fig.colorbar(im, ax=ax1)
    ax1.tick_params(axis='both', which='major', labelsize=14)
    ax1.set_title("STAT: "+ str(STAT[stat_ID]) + " -- Mw: " + str(eq_params[samp_ID,0]), fontsize=14)
    
    
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.imshow(label, vmin=vmin,vmax=vmax)
    ax2.hlines(stat_ID,0,reconstructed.shape[1],'r')
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.imshow(reconstructed,vmin=vmin,vmax=vmax)
    ax3.hlines(stat_ID,0,reconstructed.shape[1],'r')
    ax2.tick_params(axis='both', which='major', labelsize=14)
    ax3.tick_params(axis='both', which='major', labelsize=14)
    
    
    
    ax4 = fig.add_subplot(gs[2, :])
    ax4.plot(label[stat_ID,:], 'k', label="label")
    ax4.plot(noisy[stat_ID,:], 'b', label="noisy")    
    ax4.plot(reconstructed[stat_ID,:], 'r', label="recon")
    plt.legend(loc = 'lower left')
    ax4.tick_params(axis='both', which='major', labelsize=14)
    ax4.set_xlim(0,ptimes[samp_ID,stat_ID])
    
    
    # noise = np.abs(label - noisy)
    # ax5 = fig.add_subplot(gs[3, :])
    # ax5.plot(noise[stat_ID,:], 'k', label="label")
    # ax5.tick_params(axis='both', which='major', labelsize=14)
    
    
    plt.tight_layout()


modeldir = './run_db14/model_20201103-173740_400k50k/'
# modelname = "model_20200825-202813_230k60k"
modelname="model_20201103-173740_400k50k_l128"
PEGS_db_path = "../DATABASES/run_db14.hdf5"







with h5py.File("../DATABASES/REAL_DATA_NEIC_M65.hdf5", 'r') as f:
    X = np.array(f["pegs_w_noise"][:])
    labels = f["eq_params"][:]
    pwav = f["ptime"][:]
    # clip = f["clip"][:]
    X1 =np.array(f["pegs_w_noise_clip"][:])
    # X1 =np.array(f["pegs_w_noise"][:])


# Get indexes for sorting stations
stat_coords = np.loadtxt('./stations/stations_lon_lat_EFFECTIVE_FINAL.txt')
sort_lon_idx = np.argsort(stat_coords[:,0])
sort_lat_idx = np.argsort(stat_coords[:,1])





# LOAD ACTIVE stations index and sort lon/lat 
st_file = modeldir+'/'+modelname +'_stat_input.txt'
file1 = open(st_file, 'r') 
active_st_idx = []
clon = []
clat = []
STAT = []
for i, line in enumerate(file1):
    pieces = line.split()
    if np.int(pieces[6]): # This is an active station
        active_st_idx.append(i) #Save its index
    clon.append(np.float(pieces[3]))
    clat.append(np.float(pieces[4]))
    STAT.append(pieces[1])
    
file1.close()


active_st_idx = np.array(active_st_idx)
# First select the active entries, then sort
# Sorted index refers to a vector of active stations only
sort_lon_idx = np.argsort(np.array(clon)[active_st_idx])
sort_lat_idx = np.argsort(np.array(clat)[active_st_idx])


# Data parameters
t_max = 320
n_stations = len(active_st_idx) - 2 
n_comp = 3


# COMPUTE DISTANCES 
dist = DistanceMetric.get_metric('haversine')
DIST=np.zeros((labels.shape[0], len(clon) ))
R = 6371.0
# for each event
for i in range(labels.shape[0]):
    eq_lat = labels[i,1]
    eq_lon = labels[i,2]
    # for each station
    for j in range(len(clon)):
        
        temp = [[radians(eq_lat), radians(eq_lon)], [radians(np.array(clat[j])),
                                                     radians(np.array(clon[j]))]]
        DIST[i,j] = np.array(R * dist.pairwise(temp)).item(1)
         


# SELECT Real data active stations 
X1 = X1[:,active_st_idx,:,:]
# SORT REAL DATA
X1 = X1[:,sort_lon_idx,:t_max,:]
# also ptimes
ptimes =  pwav[:,active_st_idx]
ptimes= ptimes[:,sort_lon_idx]
# ALSO DIST
DIST = DIST[:,active_st_idx]

# AND STAT NAMES
STAT = np.array(STAT)[active_st_idx]
STAT = STAT[sort_lon_idx]
STAT = STAT[:n_stations]

st_lat = np.array(clat)[active_st_idx]
st_lat = st_lat[sort_lon_idx]
st_lon = np.array(clon)[active_st_idx]
st_lon = st_lon[sort_lon_idx]


# Load test data from database



batch_size = X1.shape[0]


for i in range(X1.shape[0]):
            # P-Wave arrival
        indP = np.array(np.floor(ptimes[i,:]), dtype=int)
# For each station all components
        for j in range(X1.shape[1]):
            X1[i, j, indP[j]:,:] = 0.0 


# NORMALIZATION WITH ARBITRARY SCALE
scale = 5e-9
X1 = np.clip(X1,-1.0*scale, scale)
X1 /= scale
X1 += 1.0


X1 = X1[:,:n_stations,:,:]



############## LOAD MODEL ##################
latent_dim = 128
device = torch.device('cpu')
model = VAE(n_comp, latent_dim)
PATH = modeldir+'/'+modelname+'.pth'
model.load_state_dict(torch.load(PATH, map_location=device))
model.double()
model.eval()


# Make predictions
input_data = torch.from_numpy(np.swapaxes(X1,1,-1))
recon_d, mu,logvar = model(input_data.double())
recon_D = recon_d.detach().numpy()



###############################################################################
# GEnerate synthetics corresponding to real data parameters as a reference
# for the reconstructions
sampling_rate = 1.0
sos1 = signal.iirfilter(6, 0.030 / (0.5*sampling_rate), btype='low', ftype='butter', output='sos')
sos2 = signal.iirfilter(2, 0.002 / (0.5*sampling_rate), btype='high', ftype='butter', output='sos')



# Read in parameters of all sources (lon, lat, depth, dip, strike)
samp = 1 # source spacing = 5 * samp (50km)
tmp=np.loadtxt('../Andrea_generate_database_v2.0/eq_loc/ryu_kur_lon_lat_dep_dip_str_EFFECTIVE_FINAL.txt')

dim = int(len(tmp)/samp)
EQ_PARAMS = np.zeros((dim,5))


for i in range(int(len(tmp)/samp)):
    EQ_PARAMS[i,0] = np.float(tmp[samp*i][0]) # LONGITUDE
    EQ_PARAMS[i,1] = np.float(tmp[samp*i][1]) # LATITUDE     
    EQ_PARAMS[i,2] = np.float(tmp[samp*i][2]) # DEPTH
    EQ_PARAMS[i,3] = np.float(tmp[samp*i][3]) # DIP
    EQ_PARAMS[i,4] = np.float(tmp[samp*i][4]) # STRIKE
for i in range(len(EQ_PARAMS)):
    
    if EQ_PARAMS[i,0] > 180.:
        print('Correcting lon')
        EQ_PARAMS[i,0] = EQ_PARAMS[i,0] - 360.
        



SYN_DATA0 = np.zeros(X1.shape)
SYN_PWAV = np.zeros((X1.shape[0],X1.shape[1]))
for idx_eq in range(labels.shape[0]):
    # Find closest entry in EQ+PARAMS
    idx1 = find_nearest(EQ_PARAMS[:,0], labels[idx_eq,1])
    idx2 = find_nearest(EQ_PARAMS[:,1], labels[idx_eq,2])
    
    if idx1==idx2:
        eq_k = idx1
    else:
        eq_k=idx2
        
    
    Mw = labels[idx_eq,0]
    latitude = labels[idx_eq,1]
    longitude = labels[idx_eq,2]
    depth0 = labels[idx_eq,3]
    d1 = np.abs(30-depth0)
    d2 = np.abs(20-depth0)
    if(d1<d2):
        depth = 30
    else:
        depth=20
    
    dip = EQ_PARAMS[eq_k,3]
    strike = EQ_PARAMS[eq_k,4]
    rake = np.random.normal(90,10)

    temp1, temp2 = pegslib_NEZ_one.synthetic_generator(Mw,
                                            longitude,
                                            latitude,
                                            depth,
                                            dip,
                                            strike,
                                            rake,
                                            st_lon,
                                            st_lat,
                                            sos1,
                                            sos2,
                                            filt=True)
    SYN_PWAV[idx_eq,:] = temp1[:n_stations]
    SYN_DATA0[idx_eq,:,:,:] = temp2[:n_stations,:t_max,:n_comp]
    indP = np.array(np.floor(temp1), dtype=int)
        # For each station all components
    for i in range(SYN_DATA0.shape[1]):
                     
                        
    # Set amplitudes to zero after P-wave arrival
        SYN_DATA0[idx_eq, i, indP[i]:,:] = 0.0 

            

    
    
    
    
SYN_DATA = np.swapaxes(SYN_DATA0,1,-1)
scale = 5e-9
# SYN_DATA = np.clip(SYN_DATA,-1.0*scale, scale)
# SYN_DATA /= scale
# SYN_DATA +=1.0


        
REAL_DATA = np.swapaxes(X1,1,-1)

REAL_DATA -= 1.0
REAL_DATA *= scale


recon_D -= 1.0
recon_D *= scale

# recon_DF = signal.sosfilt(sos1, recon_D, axis=2)
# recon_DF = signal.sosfilt(sos2, recon_D, axis=2)

samp_ID = 1
stat_ID = 0

plot_comparison(SYN_DATA, REAL_DATA,  recon_D, STAT, ptimes, labels, samp_ID, stat_ID)
    









































