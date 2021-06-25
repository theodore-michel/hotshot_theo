#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 12:42:05 2020

@author: licciar
"""


# These functions are used before training to split the train/va/test sets
# As this is done preprocess is applied and statistics are calculated.
# Workflow:
    # 1) 


import numpy as np
import h5py
from scipy import signal
import random
#COMPUTE STATISTICS OF TRAIN SET

def update(count, mean, M2, newValue):
#    (count, mean, M2) = existingAggregate
    count += 1
    delta = newValue - mean
    mean += delta / count
    delta2 = newValue - mean
    M2 += delta * delta2

    return (count, mean, M2)

# Retrieve the mean, variance and sample variance from an aggregate
def finalize(count, mean, M2):
#    (count, mean, M2) = existingAggregate
    if count < 2:
        return float('nan')
    else:
       (mean, variance, sampleVariance) = (mean, M2 / count, M2 / (count - 1))
       return (mean, variance, sampleVariance)



def preprocess_databases(pegs_db_path,
                         noise_db_path,
                         out_db_name,
                         idx_train,
                         idx_val,
                         idx_test,
                         active_idx, 
                         sort_idx,
                         sos1,
                         sos2):
    
    # INPUTS:
        
        # pegs_db_path
        # noise_db_path
        # idx_train
        # idx_val
        # idx_test
        # active_idx
        # sort_idx
        
        # sos1
        # sos2
        
    # Workflow:
        
        # 1) Split database of pegs in train, val, test sets
        # for each daabase:
            # 2) Select active stations and sort them as provided
            # 3) Select noise traces at random  for each entry
            # 4) Filter PEGS and NOISE using provided filters
            # 5) Add NOISE to PEGS
            # 6) Compute statistcs for train set for standardization/normalization in training
            # 7) Save dataset
            
    # OUTPUTS
    # This function saves three datasets in hdf5 format
    
    
    buffer = 1000
    t_len = 350

    ncomp = 3
    #nsamp = 3
    countt = 0
    
    
    
    ############################## CREATE EMPTY DATABASE ###################
    ntotsamp = len(idx_test) + len(idx_train) + len(idx_val)
    n_stations = len(active_idx)
    # out_db_name = "../DATABASES/run_db.hdf5"
    with h5py.File(out_db_name, 'w') as f:
    
        # Get the dimensions right and create the four datasets
        pegs_dim = ((ntotsamp, n_stations, t_len, ncomp))
        f.create_dataset("pegs_w_noise", pegs_dim, dtype = 'float64' )  
        f.create_dataset("pegs_w_noise_clip", pegs_dim, dtype = 'float64' ) 
        f.create_dataset("noise", pegs_dim, dtype = 'float64' ) 
        f.create_dataset("ptime", (ntotsamp, n_stations), dtype = 'float32' ) 
        f.create_dataset("eq_params", (ntotsamp, 5), dtype = 'float32' )  
        f.create_dataset("clip", (n_stations,ncomp,3), dtype = 'float64' )
        
    
    tkw = signal.tukey(t_len+buffer*2,0.05)
    
    for dataset in ['train', 'val', 'test']:
        
        if dataset == 'train':
            indic = idx_train
        elif dataset =='val':
            indic = idx_val
        else:
            indic = idx_test
                
    
    
        nsamp = len(indic)

        maxi = np.zeros((n_stations,ncomp,3))
        maxi[:,:,0] += 1000.0
        
        
        if (dataset == 'train'):
            massi = np.zeros(ncomp)
            mini = np.zeros(ncomp) + 10000.0
            
            meant = np.zeros(ncomp)
            M2t = np.zeros(ncomp)
            countt = np.zeros(ncomp)
            newval = np.zeros(ncomp)
            
            MEAN = np.zeros(ncomp)
            VAR = np.zeros(ncomp)
            SAMPVAR = np.zeros(ncomp)

        
        for i in range(nsamp):
            
            if(i%1000 == 0 ):
                print (dataset + ' dataset. Working on sample ' + str(i) + '/' + str(nsamp))
        
        
    ####################################### GET PEGS ####################    
            with h5py.File(pegs_db_path, 'r') as f:
                labels = np.array(f['eq_params'][indic[i],])  
                # Use the following for noise free pegs
                temp = np.array(f['pegs'][indic[i], ]) 
                X = temp[active_idx,:,:ncomp]
                temp = np.array(f['ptime'][indic[i], :])
                pwav = temp[active_idx]
    #            pwav = np.array(f['ptime'][ind,:])  
                
            # Replace first zeros with next value
            X[:,0,:] = X[:,1,:]
            # Sort
            X = X[sort_idx,:,:]  
            pwav = pwav[sort_idx]   
            
            #adjust depth label
            if(labels[4] == 30.0):
                labels[4] = 1
            else:
                labels[4] = 0

    ################################ GET NOISE AT RANDOM ####################    
            with h5py.File(noise_db_path, 'r') as f:
                tmp = f["noise_traces"]
                noise_nsamp = tmp.shape[0]
                
            noise_idx = np.random.randint (low=0, high=noise_nsamp)
            
    
            with h5py.File(noise_db_path, 'r') as f:
                temp = np.array(f["noise_traces"][noise_idx, ])
                X_noise = temp[active_idx,:,:ncomp]
                temp = np.array(f["tables"][noise_idx, ])
                noise_tables = temp[active_idx]
                noise_means = np.array(f["statistics"][noise_idx,:,:ncomp,0 ])
                clip = np.array(f["clip"][noise_idx,:,:ncomp])
                
            # Sort Stations
            X_noise = X_noise[sort_idx, :,:]
            # Sort Tables
            noise_tables = noise_tables[sort_idx]
            # Sort clip valuess
            clip = clip[sort_idx,:]
            
            
            X_noise_filt = signal.detrend(X_noise, axis=1, type='constant')
            # X_noise_filt = signal.detrend(X_noise_filt, axis=1)
            
            X_noise_filt = signal.sosfilt(sos1, X_noise,axis=1)
            X_noise_filt = signal.sosfilt(sos2, X_noise_filt,axis=1)
            
            # noise_means_filt = signal.detrend(noise_means, axis=0, type='constant')
            # noise_means_filt = signal.detrend(noise_means_filt, axis=0)
            noise_means_filt = signal.sosfilt(sos1, noise_means,axis=0)
            noise_means_filt = signal.sosfilt(sos2, noise_means_filt,axis=0)
            
            # Cut 350 seconds at random
            start_noise = np.random.randint(low=buffer, high=3600-t_len-buffer) # Index of starting noise
            t1 = start_noise #- buffer
            t2 = start_noise + t_len # + buffer
            
            noise = X_noise_filt[:, t1:t2, :]
            
            meanflag = False
            if meanflag:
                # noise = signal.detrend(noise, axis=0, type='linear')
                noise = signal.detrend(noise, axis=0, type='constant')
            
            means = noise_means_filt[t1:t2, :]
    ##########################################################################        
            
            X1 = np.zeros(X.shape)
            NOISE = np.zeros(X.shape)
            # MUting 5% of stations
            mutfac = 0.05
            num_muted = np.int(np.ceil(X.shape[0] *mutfac))
            idx_muted = random.sample(range(X.shape[0]), num_muted)
            
            # LOOP OVER STATIONS
            for j in range(X.shape[0]):
                
                # P-Wave arrival
                indP = np.array(np.floor(pwav[j]), dtype=int)
                # LOOP OVER COMPONENT
                for comp in range(ncomp):
    #
    #                    # Add noise
                        ntrace = noise[j,:,comp].copy()
    
    #                    # Check if we have only zeros for this noise segment
    #                     if not noise_tables[j]: 
    #                         # If it's a vector of zeros replace with overall mean
                            
    # #                        print(j, 'replacing zeros with mean')
    #                         # if(i==0 and comp==0):
    #                         #     with open("./log.txt","a") as f:
    #                         #         f.write(str(self.sort_stat_idx[j]) + "\n")
    #                         ntrace = means[:,comp].copy()
                            
                            
                               
                        # # ntrace = signal.detrend(ntrace)
                        # # ntrace = signal.detrend(ntrace,type='constant')
                        # # ntrace *= tkw
                        # ntrace = signal.sosfilt(sos1, ntrace)
                        # ntrace = signal.sosfilt(sos2, ntrace)
                        
                        X[j, :, comp] = signal.sosfilt(sos1, X[j,:,comp])
                        X[j, :, comp] = signal.sosfilt(sos2, X[j,:,comp])
                        
                        X[j, :, comp ]  += ntrace#[buffer : t_len+buffer]
                        
                        X1[j,:,comp] = X[j, :, comp ] 
                        NOISE[j,:,comp] = ntrace
                        
                        
                        ############### SET capping based on P-WAVE ARRIVAL ###############
                        # Set everything after P-wave arrival -2s to val
                        # Val is the value at P-wave arrival -2s
                        val = X[j, indP, comp]
                        X1[j, indP:,comp] = val
                        

                        
                        
                        # If this station has to be muted
                        if j in idx_muted:
                            X1[j, :, comp] = 0.0
                            
                        #Check if we have only zeros for this noise segment
                        if not noise_tables[j]:  
                        # # If there isn't noise data set trace to zero
                        # if all(v == 0.0 for v in ntrace):
                            X1[j, :, comp] = 0.0
                        
                        else:         
    
                            if(dataset=='train'):
                                val = np.std(ntrace)#[buffer : t_len+buffer])
                                # Store std for mean calculations
                                
                                maxi[j,comp,2] += val
                                # Find minimum std 
                                if val<maxi[j,comp,0]:
                                    maxi[j,comp,0] = val
                                # Fins maximum std
                                if val>maxi[j,comp,1]:
                                    maxi[j,comp,1] = val
                                
                                
                                

                                
                            
                                    
                                
    #        

    
                        # # Cap amplitudes
                        # cap = 10.0 * clip[j,comp] #np.std(ntrace[buffer:self.dim2+buffer])
                        
                        # for k in range(X.shape[1]):
                        #     val = X[j, k, comp]
                            
                        #     if (np.abs(val) > cap):
                        #         X[j, k:, comp ] = cap
                        #         if (val>0.0):
                        #             X[j, k:, comp ] = cap
                        #         else:
                        #             X[j, k:, comp ] = -1.0 * cap
                        #         break
                            
        
                            
            # # COmpute statistics of trainset per channel
            # if (dataset == 'train'):
            #     valmax = X1.max()
            #     valmin = X1.min() 
            #     if valmax > massi:
            #         massi = valmax
            #     if valmin < mini:
            #         mini = valmin
                
                # # COmpute running mean and variance 
                # if i == 0:
                #     meant = X1.copy()
                #     M2t = np.zeros((X.shape))
                #     countt = 1
                # else:    
                #     newval = X1.copy()
                #     countt, meant, M2t = update(countt, meant, M2t, newval)            
    
                
                    # COmpute running mean and variance 
                    # if i == 0:
                    #     meant[comp] = X1.mean(axis=(0,1))[comp]
                    #     M2t[comp] = 0.0#X1.var(axis=(0,1))[comp]
                    #     countt[comp] = 1
                    # else:    
                    #     newval[comp] = X1.mean(axis=(0,1))[comp]
                    #     countt[comp], meant[comp], M2t[comp] = update(countt[comp],
                    #                                                  meant[comp],
                    #                                                  M2t[comp], 
                    #                                                  newval[comp]) 

                        
                    # print ("Computing stats for sample: " + str(i) + "and comp: " + str(comp) )
                    # print("massi[comp]: " + str(massi[comp]))
                    # print("mini[comp]: " + str(mini[comp]) )
                    # print("meant[comp]: " + str(meant[comp]) )
                    # print("count[comp]: " + str(countt[comp]) )
            # FILL NEW DATABASE
            with h5py.File(out_db_name, 'a') as f:
                # SAVE EQ PARAMS
                f["eq_params"][indic[i], ] = labels
                f["pegs_w_noise"][indic[i], ] = X
                f["pegs_w_noise_clip"][indic[i], ] = X1
                f["noise"][indic[i], ] = NOISE
                f["ptime"][indic[i], ] = pwav 
            
            if (dataset == 'train'): 
                for comp in range(ncomp):
                    # COmpute statistics of trainset per channel
                    # Update  min, max
                    valmax = X1.max(axis=(0,1))[comp]
                    valmin = X1.min(axis=(0,1))[comp] 
                    if valmax > massi[comp]:
                        massi[comp] = valmax
                    if valmin < mini[comp]:
                        mini[comp] = valmin   
                        
                if(i==0):
                    CUM1 = np.zeros(X.shape)
                    CUM2 = np.zeros(X.shape)
                    CUM3 = 0.0
                    CUM4 = 0.0
                    N = nsamp #* X.size/3
                    N1 = N * X.size/3
                CUM3 += X1.sum(axis=(0,1))
                CUM4 += np.sum(X1**2, axis=(0,1))
                CUM1 += X1
                CUM2 += X1**2               
                
        

        
        
        # if (dataset=='train'):
        #     MEAN[0], VAR[0], SAMPVAR[0] = finalize (countt[0],meant[0],M2t[0])
            
        #     MEAN[1], VAR[1], SAMPVAR[1] = finalize (countt[1],meant[1],M2t[1])
            
        #     MEAN[2], VAR[2], SAMPVAR[2] = finalize (countt[2],meant[2],M2t[2])
            
 
            # print ("Finalized mean for comp: 0 " + str(MEAN[0] ))
            # print ("Finalized mean for comp: 1 " + str(MEAN[1] ))
            # print ("Finalized mean for comp: 2 " + str(MEAN[2] ))
        if (dataset=='train'):            
            # Compute mean std for each station and component 
            maxi[:,:,2] /= nsamp
            
            with h5py.File(out_db_name,'a') as f:
                # Save clipping values
                f["clip"][:] = maxi
    
    # return MEAN, np.sqrt(SAMPVAR), mini, massi 
    # N = X.size/ncomp
    MEAN_image = CUM1/N
    VAR_image =  1.0/(N*(N-1)) * (N* CUM2 - (CUM1)**2)
    MEAN = CUM3/N1
    VAR=  1.0/(N1*(N1-1)) * (N1* CUM4 - (CUM3)**2)
    return MEAN_image, np.sqrt(VAR_image), MEAN, np.sqrt(VAR), mini, massi 

    
    
    
    
    
    
    
    
    
    
    
#    
#    
#    
#    
#    
#    
#    
#    
#    nsamp = len(list_train)   
#    # We can't load the whole database at once so I work with batches
#    nbatch = np.int(nsamp/batchsize)
#    
#    
#    massi = 0.0
#    mini = 1000.0
#    
#    for i in range(nbatch):
#        
#        print ('Working on batch ' + str(i+1) + '/' + str(nbatch))
#        ind0 = i*batchsize
#        ind1 = (i+1)*batchsize
#    #    print (ind0, ind1)
#    
#        # Load data
#        with h5py.File(databasepath,'r') as f:
#            X = f['pegs'][ind0:ind1]
#    #        X1 = f['pegs_w_noise'][ind0:ind1]
#        valmax = X.max()
#        valmin = X.min() 
#        if valmax > massi:
#            massi = valmax
#        if valmin < mini:
#            mini = valmin
#            
#        if i == 0:
#            meant = np.mean(X)
#            countt= 1
#            M2t = 0.0
#        else:    
#            newval = np.mean(X)
#            countt, meant, M2t = update(countt, meant, M2t, newval)
#    
#    MEAN, VAR, SAMPVAR = finalize(countt, meant, M2t)
#    
#    return MEAN, np.sqrt(VAR) , mini, massi 

