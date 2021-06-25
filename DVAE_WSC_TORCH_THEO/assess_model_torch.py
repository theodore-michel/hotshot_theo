#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 21 13:33:33 2020

@author: licciar
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
# from Classes import HDF5Dataset, VAE
from Classes import *
from scipy import signal
import h5py
import pickle
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.gridspec import GridSpec






def plot_comparison(label,
                    noisy,
                    reconstructed, 
                    ptimes,
                    eq_params, 
                    samp_ID, 
                    stat_ID):
    
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
    
    vmin = 0.6
    vmax = 1.1
    
    ########################
    fig = plt.figure(constrained_layout=False)
    gs = GridSpec(4, 2, figure=fig)
    ax1 = fig.add_subplot(gs[0,:])
    im = plt.imshow(noisy, vmin=vmin,vmax=vmax)
    fig.colorbar(im, ax=ax1)
    ax1.tick_params(axis='both', which='major', labelsize=14)
    ax1.set_title("Mw: " + str(eq_params[samp_ID,0]), fontsize=14)
    
    vmin = 0.9
    vmax = 1.1
    ax2 = fig.add_subplot(gs[1, 0])
    im = plt.imshow(label, vmin=vmin,vmax=vmax)
    fig.colorbar(im, ax=ax2)
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.imshow(reconstructed,vmin=vmin,vmax=vmax)
    ax2.tick_params(axis='both', which='major', labelsize=14)
    ax3.tick_params(axis='both', which='major', labelsize=14)
    
    
    
    ax4 = fig.add_subplot(gs[2, :])
    ax4.plot(label[stat_ID,:], 'k', label="label")
    ax4.plot(noisy[stat_ID,:], 'b', label="noisy")    
    ax4.plot(reconstructed[stat_ID,:], 'r', label="recon")
    plt.legend(loc = 'lower right')
    ax4.tick_params(axis='both', which='major', labelsize=14)
    
    
    
    noise = np.abs(label - noisy)
    ax5 = fig.add_subplot(gs[3, :])
    ax5.plot(noise[stat_ID,:], 'k', label="label")
    ax5.tick_params(axis='both', which='major', labelsize=14)
    
    
    plt.tight_layout()
    
    # rangel=np.zeros((2,2))
    # rangel[0,0] = y_Mw.min()
    # rangel[0,1] = y_Mw.max()
    # rangel[1,0] = y_Mw.min()
    # rangel[1,1] = y_Mw.max()
    # # COmpute density plot
    # #sns.kdeplot(y_Mw, preds[:,0], cmap="Blues", shade=False, shade_lowest=False)
    # h = ax.hist2d(y_Mw, preds[:,0], bins=(100, 100), vmin=mincount, vmax=maxcount)#,cmap=plt.cm.jet)
    # fig.colorbar(h[3], ax=ax)
    # # plt.clim(0,125)
    # ax.plot([6.5,10.0],[6.5,10.0],'r')
    # # ax.axis('equal')
    # ax.set_xlim(6.5,10)
    # ax.set_ylim(6.5,10)
    # ax.tick_params(axis='both', which='major', labelsize=14)
    
    # # Major ticks every 0.5, minor ticks every 0.25
    # major_ticks = np.arange(6.5, 10.0, 0.5)
    # minor_ticks = np.arange(6.5, 10.0, 0.25)
    
    # ax.set_xticks(major_ticks)
    # ax.set_xticks(minor_ticks, minor=True)
    # ax.set_yticks(major_ticks)
    # ax.set_yticks(minor_ticks, minor=True)
    
    # #ax.plot(y_Mw, preds[:,0], 'or')
    # ax.set_title('Mw',fontsize=14)
    # ax.set_xlabel('Mw True',fontsize=14)
    # ax.set_ylabel('Mw Pred',fontsize=14)
    # ax.grid(which='both', axis='both', linestyle='--')
    # ax = fig.add_subplot(2, 3, 4)
    # ax.hist(resid,bins=nbins, range=(res_low,res_high), density=True)
    # ax.axis((res_low,res_high, 0, 2))
    # ax.set_xlabel('Mw Residuals',fontsize=14)
    # ax.tick_params(axis='both', which='major', labelsize=14)
    # ax.text(res_low,-0.4, 'MSE = %7.3f  mean = %7.3f  sd= %7.3f' %(mse, meana,sta), fontsize=14)
    




# Load Existing model



modeldir = './run_db14/model_20201103-173740_400k50k/'
# modelname = "model_20200825-202813_230k60k"
modelname="model_20201103-173740_400k50k_l128"
PEGS_db_path = "../DATABASES/run_db14.hdf5"




test_idx = np.loadtxt(modeldir+'/'+modelname+'_test_idx.txt')
test_idx = np.sort(test_idx)
test_ids= list(str(int(w)) for w in test_idx)




val_idx = np.loadtxt(modeldir+'/'+modelname+'_val_idx.txt')
val_idx = np.sort(val_idx)
val_ids= list(str(int(w)) for w in val_idx)

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
for i, line in enumerate(file1):
    pieces = line.split()
    if np.int(pieces[6]): # This is an active station
        active_st_idx.append(i) #Save its index
    clon.append(np.float(pieces[3]))
    clat.append(np.float(pieces[4]))
    
file1.close()

active_st_idx = np.array(active_st_idx)
# First select the active entries, then sort
# Sorted index refers to a vector of active stations only
sort_lon_idx = np.argsort(np.array(clon)[active_st_idx])
sort_lat_idx = np.argsort(np.array(clat)[active_st_idx])

# Load test data from database
# Data parameters
t_max = 320
n_stations = len(active_st_idx) -2
n_comp = 3
#LAtent dim is part of modelname. Extract it!
latent_dim = 128


############## LOAD MODEL ##################
device = torch.device('cpu')
# model = VAE(n_comp, latent_dim)
model = DVAE_WSC(n_comp, latent_dim,skips=[True,True,True])
PATH = modeldir+'/'+modelname+'.pth'
model.load_state_dict(torch.load(PATH, map_location=device))
model.double()
model.eval()



##################  HERE WE DEFINE THE GENERATOR
# Generate batch_size samples from test and val set
batch_size = 32


#print(train_ids)
working_db = PEGS_db_path[:-5]

# Parameters
params = {'batch_size': batch_size,
          'shuffle': False,
          'num_workers': 1}

# Datasets

partition = { 
    "validation": val_idx,
    "test" : test_idx
    }# IDs


# Generators
test_set = HDF5Dataset(partition['test'], 
                           n_stations, 
                           t_max, 
                           n_comp,
                           database_path=working_db+".hdf5"
                           )
test_loader = torch.utils.data.DataLoader(test_set, **params)

validation_set = HDF5Dataset(partition['validation'], 
                             n_stations, 
                             t_max, 
                             n_comp,
                             database_path=working_db+".hdf5" 
                             )
val_loader = torch.utils.data.DataLoader(validation_set, **params)



# EXTRACT ACTUAL SAMPLES
# data_n, data, X, label, noise =iter(test_loader).next()
data_n, data, ptimes, eq_params =iter(test_loader).next()
recon_d, mu,logvar = model(data_n.double())


# PLOTTABLE NUMPY THINGS AFTER RESCALING
data_N = data_n.numpy()
data_F = data.numpy()
# X = X.numpy()
# label = label.numpy()
# noise = noise.numpy()
recon_D = recon_d.detach().numpy()

ptimes = ptimes.numpy()
eq_params = eq_params.numpy()

# scale = 5e-9
# data_N -= 1.0
# data_N *= scale
# data_F -= 1.0
# data_F *= scale

# recon_D -= 1.0
# recon_D *= scale




# PLOT A comparison between a noisy input data and a reconstructed image
ID_samp = 12
ID_stat = 22
plot_comparison(data_F, data_N, recon_D, ptimes,eq_params, ID_samp , ID_stat )






###################### PLOT LEARNING HISTORY
val_loss = np.loadtxt(modeldir+'/'+modelname+'_val_losses.txt')
train_loss = np.loadtxt(modeldir+'/'+modelname+'_train_losses.txt')

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
epochs = np.arange(len(val_loss))
# ax.semilogy(epochs, train_loss, label='Train')
# ax.semilogy(epochs, val_loss, label='Val')
ax.plot(epochs, train_loss, label='Train')
ax.plot(epochs, val_loss, label='Val')
# ax.set_xlabel('Epochs',fontsize=14)
ax.set_ylabel('Loss (mae)',fontsize=14)
ax.set_title('Total loss',fontsize=14)
ax.grid(which='both', axis='both')
ax.tick_params(axis='both', which='major', labelsize=14)
ax.legend()
#ax.grid(True)



# ax = fig.add_subplot(2, 2, 4)
# ax.semilogy(epochs,dicto['d_output_acc'], label='Train')
# ax.semilogy(epochs, dicto['val_d_output_acc'], label='Val')
# # ax.plot(epochs,dicto['d_output_acc'], label='Train')
# # ax.plot(epochs, dicto['val_d_output_acc'], label='Val')
# ax.set_xlabel('Epochs',fontsize=14)
# ax.set_ylabel('Accuracy',fontsize=14)
# ax.set_title('Depth',fontsize=14)
# ax.grid(which='both', axis='both')
# ax.tick_params(axis='both', which='major', labelsize=14)
# ax.legend()

fig.set_size_inches((12.0, 12.0), forward=False)
fig.savefig(modeldir+ '/' +modelname+"_histories.png", dpi=300)


