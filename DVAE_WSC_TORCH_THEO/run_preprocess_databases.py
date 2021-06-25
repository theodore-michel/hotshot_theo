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
import os, argparse




def preprocess_databases(pegs_db_path,
                         noise_db_path,
                         out_db_name,
                         idx_train,
                         idx_val,
                         idx_test,
                         ncomp,
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
            # 6) Save dataset
            
    # OUTPUTS
    # This function saves three datasets in hdf5 format
    
    
    bufferleft = 1000
    bufferright = 250
    t_len = 350

    # ncomp = 1

    
    
    ############################## CREATE EMPTY DATABASE ###################
    ntotsamp = len(idx_test) + len(idx_train) + len(idx_val)
    n_stations = len(active_idx)
    # out_db_name = "../DATABASES/run_db.hdf5"
    with h5py.File(out_db_name, 'w') as f:
    
        # Get the dimensions right and create the four datasets
        pegs_dim = ((ntotsamp, n_stations, t_len*2, ncomp))
        # f.create_dataset("pegs_w_noise", pegs_dim, dtype = 'float64' )  
        f.create_dataset("data", pegs_dim, dtype = 'float64' ) 
        f.create_dataset("label", pegs_dim, dtype = 'float64' ) 
        f.create_dataset("STF", (ntotsamp,t_len), dtype = 'float64')
        f.create_dataset("ptime", (ntotsamp, n_stations), dtype = 'float32' ) 
        f.create_dataset("eq_params", (ntotsamp, 6), dtype = 'float32' )  
        # f.create_dataset("clip", (n_stations,ncomp,3), dtype = 'float64' )
        
    # Create a list of noise idx
    with h5py.File(noise_db_path, 'r') as f:
        tmp = f["noise_traces"]
        noise_nsamp = tmp.shape[0]
    # Create the list of ids
    list_idx_noise = np.arange(noise_nsamp)
    #Shuffle the list (in place)
    np.random.shuffle(list_idx_noise)
    # Preprocess the datasets
    for dataset in ['train', 'val', 'test']:
        
        if dataset == 'train':
            indic = idx_train
            id_noise = list_idx_noise[0:4510]
        elif dataset =='val':
            indic = idx_val
            id_noise = list_idx_noise[4510:5798]
        else:
            indic = idx_test
            id_noise = list_idx_noise[5798:]
                
    
    
        nsamp = len(indic)        
        

        
        for i in range(nsamp):
            
            # For this sample choose if it's noise (0) or pegs(1)
            # pegs_choice = random.choices([0,1], [0.3,0.7])[0]
            # Always PEGS for the moment
            pegs_choice = 1
            
            if pegs_choice:
                stringa = 'PEGS'
            else:
                stringa = 'NOISE'
            
            if(i%1000 == 0 ):
                # print (dataset + ' dataset. Working on sample ' + str(i) + '/' + str(nsamp))
                print ('{} dataset. Working on sample {}/{}. Selected data is {}'.format(dataset,
                                                                                         i,
                                                                                         nsamp,
                                                                                         stringa)  )     
    ####################################### GET PEGS ####################    
            with h5py.File(pegs_db_path, 'r') as f:
                labels = np.array(f['eq_params'][indic[i],])  
                STF = np.array(f['moment_rate'][indic[i],])
                # Use the following for noise free pegs
                temp = np.array(f['pegs'][indic[i], ]) 
                X = temp[active_idx,:,:ncomp]
                temp = np.array(f['ptime'][indic[i], :])
                pwav = temp[active_idx]

                          
            
            X = X[sort_idx,:,:]  
                
            # Now filter PEGS
            X= signal.sosfilt(sos1, X,axis=1)
            X= signal.sosfilt(sos2, X,axis=1)
                        
            pwav = pwav[sort_idx]   
            
            #adjust depth label
            if(labels[4] == 30.0):
                labels[4] = 1
            else:
                labels[4] = 0
            
            eq_params = np.zeros(6)
            eq_params[:-1] = labels.copy() 

    ################################ GET NOISE AT RANDOM ####################    
            with h5py.File(noise_db_path, 'r') as f:
                tmp = f["noise_traces"]


            noise_idx = np.random.choice(id_noise)
            
            
    
            with h5py.File(noise_db_path, 'r') as f:
                temp = np.array(f["noise_traces"][noise_idx, ])
                X_noise = temp[active_idx,:,:ncomp]
                temp = np.array(f["tables"][noise_idx, ])
                noise_tables = temp[active_idx]
                # noise_means = np.array(f["statistics"][noise_idx,:,:ncomp,0 ])
                # clip = np.array(f["clip"][noise_idx,:,:ncomp])
                
            # Sort Stations
            X_noise = X_noise[sort_idx, :,:]
            # Detrend, Demean , filterer
            X_noise_filt = X_noise.copy()
            # X_noise_filt = signal.detrend(X_noise_filt, axis=1, type='linear')
            # X_noise_filt = signal.detrend(X_noise_filt, axis=1, type='constant')
            X_noise_filt = signal.sosfilt(sos1, X_noise_filt,axis=1)
            X_noise_filt = signal.sosfilt(sos2, X_noise_filt,axis=1)

            # Sort Tables
            noise_tables = noise_tables[sort_idx]
            # Sort clip valuess
            # clip = clip[sort_idx,:]
            
            
            # Cut 700 seconds at random
            start_noise = np.random.randint(low=bufferleft, high=3600-700-bufferright) # Index of starting noise
         
        
            t1 = start_noise #- buffer
            t2 = start_noise + 700 # + buffer
            if(i%1000 == 0 ):
                print ("Noise sample idx {} -- start {} end {}".format(noise_idx,
                                                                   t1,t2))
            
            X_noise_filt_cut = X_noise_filt[:,t1:t2,:].copy()
            # X_noise_filt_cut = signal.detrend(X_noise_filt_cut, axis=1, type='linear')
            # X_noise_filt_cut = signal.detrend(X_noise_filt_cut, axis=1, type='constant')
    ##########################################################################        
            
            X1 = np.zeros((X.shape[0],700,X.shape[2]))
            X_label = np.zeros(X1.shape)
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
                    
                        # Check if this is in noise tables
                        #Check if we have only zeros for this noise segment
                        if not noise_tables[j]:  
                        # # If there isn't noise data set trace to zero for all components
                        # and go to next station
                            X1[j, :, :] = 0.0
                            X_label[j,:,:] = 0.0
                            break

                            

                        
                        # First noise

                        ntrace = X_noise_filt_cut[j,:,comp].copy()


                        if np.max(np.abs(ntrace)) < 8.0e-12:
                            # print("culprit")
                            X1[j, :, :] = 0.0
                            X_label[j,:,:] = 0.0
                            break
                        
                        # MAke a choice if this is pegs+noise or only noise
                        
                        # If choice is 1 with 0.7 probability save pegs
                        if pegs_choice:
                        
                            
                            
                            # Add noise trace 
                            X1[j,X.shape[1]:,comp] = X[j, :, comp ].copy()
                            X1[j, :, comp ]  += ntrace.copy()
                            
                            # Save label (noise free pegs) 
                            X_label[j,X.shape[1]:,comp] = X[j, :, comp ].copy()
                            

                            
                            # Save_eq_params flagging pegs
                            eq_params[-1] = 1.0
                                
                            
                        # If choice is zero, only save the noise trace 
                        else:
                            
                            # Noise trace into "data" 
                            X1[j,:,comp] = ntrace.copy()
                            # Be sure that the label for this is zeros
                            X_label[j,:,comp] = 0.0 
                        
                            # Save eq_params flagging noise 
                            eq_params[-1] = 0.0
                            
                        
                        ############### SET trace to zero after P-WAVE ARRIVAL ###############
                        # Set everything after P-wave arrival to zero
                                     
                        X1[j, indP+X.shape[1]:,comp] = 0.0
                        X_label[j, indP+X.shape[1]:,comp] = 0.0   
                            
                        # If this station has to be muted
                        if j in idx_muted:
                            X1[j, :, comp] = 0.0
                            X_label[j,:,comp] = 0.0

 

            # Out of station loop
            # FILL NEW DATABASE
            with h5py.File(out_db_name, 'a') as f:
                # SAVE EQ PARAMS
                f["eq_params"][indic[i], ] = eq_params
                f["STF"][indic[i]] = np.cumsum(STF)
                f["data"][indic[i], ] = X1
                f["label"][indic[i], ] = X_label
                f["ptime"][indic[i], ] = pwav 


    return idx_muted

"""parsing and configuration"""
def parse_args():
    desc = "Preprocess data for PEGSNET"
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('--train_n_samp', type=int, default=100,                   
                        help='Number of examples in training set')
    
    parser.add_argument('--val_n_samp', type=int, default=20,                   
                        help='Number of examples in validation set')
    
    parser.add_argument('--test_n_samp', type=int, default=20,                   
                        help='number of examples in test set')
    
    parser.add_argument('--ncomp', type=int, default=1,                   
                        help='number of seismic components to use.')
    
    parser.add_argument('--PEGS_db_path', type=str, 
                    default='../DATABASES/database_NEZ_no_filt_no_noise_new_55_100_500k.hdf5',
                        help='PATH to PEGS database')
    
    parser.add_argument('--NOISE_db_path', type=str, default='../DATABASES/noise_db6.hdf5',
                        help='PATH to NOISE database')
    
    parser.add_argument('--output_db', type=str, default='../DATABASES/run_db99',
                        help='Output database path and name (prefix)')
    
    parser.add_argument('--sorting', type=str, default='lon', choices=['lon', 'lat', None],
                        help='How to sort stations')
    
    parser.add_argument('--low_freq', type=float, default=0.002,
                        help='Lower frequency for highpass')
    
    parser.add_argument('--high_freq', type=float, default=0.03,
                        help='Higher frequency for lowpass')


    return check_args(parser.parse_args())


"""checking arguments"""
def check_args(args):
    #   --PEGS_db_path       
    if not os.path.exists(args.PEGS_db_path):
        print(args.PEGS_db_path + ' PEGS path does not exist')
        exit()

    #   --NOISE_db_path       
    if not os.path.exists(args.NOISE_db_path):
        print(args.NOISE_db_path + ' NOISE path does not exist')
        exit()


    # frequencies
    try:
        assert args.low_freq < args.high_freq
    except:
        print('low_freq must be smaller than high_freq')

    # number of samples
    # try:
    #     assert args.train_n_samp >= args.val_nsamp
    # except:
    #     print('batch size must be larger than or equal to one')

    return args

def main():
    
    
        # parse arguments
    args = parse_args()
    if args is None:
        exit()

    np.random.seed(42) 
    random.seed(42)
    PEGS_db_path = args.PEGS_db_path
    NOISE_db_path = args.NOISE_db_path
    working_db = args.output_db
    f_low = args.low_freq
    f_high = args.high_freq
    ncomp =args.ncomp
    
    
    
    # LOAD ACTIVE stations index and sort lon/lat 
    st_file = './stations/stat_input.txt'
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
    if args.sorting == 'lon':
        sort_idx = np.argsort(np.array(clon)[active_st_idx])
    elif args.sortin =='lat':
        sort_idx = np.argsort(np.array(clat)[active_st_idx])
    else:
        sort_idx = np.arange(0,len(active_st_idx))
        
    

    
    
    
    
    ##################  Parameters of train/val/test sets ####################
    train_nsamp = args.train_n_samp
    val_nsamp = args.val_n_samp
    test_nsamp = args.test_n_samp
    tot_samp = train_nsamp + val_nsamp + test_nsamp
    
    # Create the list of ids
    temp = np.arange(tot_samp)
    #Shuffle the list (in place)
    np.random.shuffle(temp)
    
    list_train = temp[:train_nsamp]
    list_val = temp[train_nsamp:train_nsamp+val_nsamp]
    list_test = temp[train_nsamp+val_nsamp:]
    
    np.save(working_db+"_train_idx",list_train)
    np.save(working_db+"_val_idx",list_val)
    np.save(working_db+"_test_idx",list_test)
    
    
    sampling_rate = 1.0
    sos1 = signal.iirfilter(6, f_high / (0.5*sampling_rate), btype='low', ftype='butter', output='sos')
    sos2 = signal.iirfilter(2, f_low / (0.5*sampling_rate), btype='high', ftype='butter', output='sos')
        
    null = preprocess_databases(PEGS_db_path,
                               NOISE_db_path,
                                working_db+'.hdf5',
                                list_train,
                                list_val,
                                list_test,
                                ncomp,
                                active_st_idx, 
                                sort_idx,
                                sos1,
                                sos2)



if __name__ == "__main__":
   main()

    
    
    
    
