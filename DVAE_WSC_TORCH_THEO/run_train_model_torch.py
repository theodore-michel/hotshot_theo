#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 11 13:24:14 2020

@author: andrea
"""

import os, shutil, argparse, sys
from time import time
import torch
from torch import  optim
from Classes import HDF5Dataset, DAE, DistributedEvalSampler, DVAE_WSC
import matplotlib
from visuals import *
import random
import numpy as np
#from torchsummary import summary
from datetime import datetime
from torch.nn import functional as F

from torchvision.utils import save_image
from torch import nn
import subprocess
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import idr_torch
import torch.multiprocessing as mp

# Use the same seed for testing
#torch.manual_seed(42)
#torch.backends.cudnn.deterministic = True
#torch.backends.cudnn.benchmark = False
#np.random.seed(42)

torch.autograd.set_detect_anomaly(True)

def setup(PROCESS_RANK,WORLD_SIZE):

    # initialize the process group
    dist.init_process_group("nccl", init_method='env://',rank=PROCESS_RANK, world_size=WORLD_SIZE)

def cleanup():
    dist.destroy_process_group()

def set_backend(backend='agg'):
    # since Jean-Zay doesn't support interactive backend, set to non-interactive 
    # backend like 'agg'. For Hotshot: set back to 'qt5agg'
    matplotlib.use(backend)


"""parsing and configuration"""
def parse_args():
    desc = "Calculate statistics of a given dataset"
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('--db_name', type=str, default='../DATABASES/run_db_test',                   
                        help='Path of the preprocessed database (without .hdf5)')
    
    
    parser.add_argument('--model_outname', type=str, default='test',                   
                        help='Prefix name for output model. A subdirectory under ./outputs will be created)')
    
    parser.add_argument('--train_nsamp', type=int, default=500, 
                        help='Number of examples for the training set.')
    
    parser.add_argument('--val_nsamp', type=int, default=50, 
                        help='Number of examples for the validation set.') 
    
    parser.add_argument('--test_nsamp', type=int, default=50, 
                        help='Number of examples for the test set.')  
    
    parser.add_argument('--batch_size', type=int, default=128, 
                        help='The size of the batches.')  

    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate.')

    parser.add_argument('--ncomp', type=int, default=3,
                        help='Number of seismic components to use.')

    parser.add_argument('--skip_co', type=str, default='111',   # change if model structure changes
                        help='Which skip connections to use.')
    
    parser.add_argument('--n_epochs', type=int, default=100, 
                        help='Number of epochs.')  
    
    parser.add_argument('--log_interval', type=int, default=100, 
                        help='Print info every log_interval batches.') 
    
    parser.add_argument('--latent_dim', type=int, default=128, 
                        help='Dimension of the bottleneck.') 
    
    parser.add_argument('--check_interval', type=int, default=20,
                        help='Save a checkpoint every check_interval epochs.')

    parser.add_argument('--resume_chkp', type=str, default=None,
                        help='Resume training from the specified checkpoint path.')
    
    parser.add_argument('--platform', type=str, default='hotshot', 
                        choices=['laptop', 'hotshot', 'JZ'],
                        help='Choose where to run training. Mostly controls how to dela with gpus. Need to modify ./idr_torch.py accordingly.')

    parser.add_argument('--test_mode', type=str, default='BEST', choices=['BEST', 'LAST'],
                        help='Use BEST (lower validation loss) or LAST model for predictions on test set.')
    
    parser.add_argument('--loss', type=str, default='BCE', choices=['BCE', 'MSE'],
                        help='Use BCE or MSE loss for reconstruction loss in network loss function.')


    return check_args(parser.parse_args())


"""checking arguments"""
def check_args(args):
    #   --PEGS_db_path       
    if not os.path.exists(args.db_name+'.hdf5'):
        print(args.db_name + '.hdf5 does not exist')
        exit()
    if len(args.skip_co)>3: # change if model structure changes
        print("{} skip connections specified, only 3 possible.".format(len(args.skip_co)))
        exit()
    return args

def initialize_weights(m):
    if isinstance(m, nn.Conv2d):
      nn.init.xavier_uniform_(m.weight)
      if m.bias is not None:
          nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Linear):
      nn.init.kaiming_uniform_(m.weight)
      nn.init.constant_(m.bias, 0)
    # read in Goyal et al. (2018) that setting batchnorm to 0 at end of residuals
    # improves model by 0.2-0.3% ... (https://arxiv.org/abs/1706.02677)


# def enable_dropout(model):
#     """ Function to enable the dropdistriout layers during test-time """
#     for m in model.modules():
#         if m.__class__.__name__.startswith('Dropout'):
#             print("switching dropout on " + str(m.__class__.__name__))
#             m.train()    



# Reconstruction + KL divergence losses summed over all elements and batch
# maybe improve this loss function to be more specific to our case? 

def smoothness(images, h_coeff, v_coeff):       # supposes (N,C,H,W) format for output
    # if more smoothness wanted between stations then increase v_coeff
    # if more smoothness wanted through time for each station then increase h_coeff
    
    # batch_n = images.shape[0]
    # channel = images.shape[1]
    height  = images.shape[2]
    width   = images.shape[3]
    
    horizontal_normal = torch.narrow(images, dim=2, start=0, length=height-1)    # take first H-1 rows of tensor
    horizontal_shift  = torch.narrow(images, dim=2, start=1, length=height-1)    # take last H-1 rows of tensor
    
    vertical_normal = torch.narrow(images, dim=3, start=0, length=width-1)       # take first W-1 columns of tensor
    vertical_shift  = torch.narrow(images, dim=3, start=1, length=width-1)       # take last W-1 columns of tensor
    
    horizontal = torch.pow(horizontal_normal-horizontal_shift, 2)
    vertical   = torch.pow(vertical_normal-vertical_shift, 2)
    
    h_loss = torch.mean(horizontal)
    v_loss = torch.mean(vertical)
    
    return h_loss*h_coeff + v_loss*v_coeff

def loss_function(recon_x, x, mu, logvar,bceCoeff=1, kldCoeff=1, slCoeff=1, recArg = 'BCE'):
    # reconstruction loss:
    if recArg=='BCE': recon_loss  = bceCoeff * F.binary_cross_entropy(recon_x, x, reduction='sum') # /(320*72) for avg over all batch images 
    if recArg=='MSE': recon_loss  = bceCoeff * torch.mean( torch.pow((recon_x - x),2) ) #recommended bceCoeff=1e3, kldCoeff=1e-3 1e-4
    # smoothness loss:
    smooth_loss = slCoeff * smoothness(recon_x, h_coeff=1, v_coeff=2) # more important that signal be smooth through time (not space)
        
    # latent loss :
    latent_loss = kldCoeff * (-0.5) * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    # return BCE, KLD, SL
    return recon_loss, latent_loss, smooth_loss


def predict_test_set(Dmodel, modelname, db_name, shape, test_mode, distributed, WORLD_SIZE, LOCAL_RANK, PROCESS_RANK):

    # Get indexes of test set
    test_idx = np.loadtxt(modelname + '_test_idx.txt')
    # Get total number of examples in the test set
    numex = len(test_idx)
    # Sort them (not needed)
    test_idx = np.sort(test_idx)
    # Get the indexes
    test_indici = test_idx[:numex]
    # Number of batch samples for each gpu
    batch_size_per_gpu = 500
    # LOAD THE RELEVANT MODEL
    if test_mode == 'BEST':

        MODELPATH = modelname + "_BEST"

    elif test_mode == 'LAST':

        MODELPATH = modelname 

    gpu = torch.device("cuda")

    if distributed:

        ################### NOW LOAD IT EVERYWHERE
        #Synchronize on all gpus
        dist.barrier()
        map_location = {'cuda:%d' % 0: 'cuda:%d' % LOCAL_RANK} 
        checkpoint = torch.load(MODELPATH+"_DDP.pth", map_location=map_location)
        Dmodel.load_state_dict(checkpoint) # load model
        # Each process (GPU) will save tot_samples/4 examples. So divide the lenght of test set by four
        samp_per_gpu = int(len(test_indici)/WORLD_SIZE)


    else:
        checkpoint = torch.load(MODELPATH + ".pth")
        Dmodel.load_state_dict(checkpoint) # load last model
        samp_per_gpu = len(test_indici)

    Dmodel.eval() 

    ############ DEFINE TEST LOADER
    # Parameters
    if distributed:
        params = {'batch_size': batch_size_per_gpu,
                  'shuffle': False,
                  'num_workers': 40,
                  'pin_memory':True}
    else:
        params = {'batch_size': batch_size_per_gpu,
          'shuffle': True,
          'num_workers': 16,
          'pin_memory':True}

    # Shape is a tuple with n_stations, t_max, ncomp
    # For each time step make predictions on a batch
    # for s in range(shape[1]+1):
    #     if PROCESS_RANK ==0:
    #         print('Processing time_step: ' + str(i) + '/' +str(shape[1]+1))

    # For the moment we work with no-time_stepping and shift = 0
    s = 0

    # Initialize output matrices
    TRUE = np.zeros((samp_per_gpu, shape[1], shape[0], 2))
    PREDS = np.zeros((samp_per_gpu, shape[1], shape[0]))
    EQ_PAR = np.zeros((samp_per_gpu,3))


    test_set = HDF5Dataset(test_indici, 
                                   shape[0], 
                                   shape[1], 
                                   shape[2],
                                   dshift = s,
                                   database_path=db_name+".hdf5"
                                   )

    if distributed:

        test_sampler = DistributedEvalSampler(test_set, num_replicas=WORLD_SIZE,
                                                          rank=PROCESS_RANK)

        test_loader = torch.utils.data.DataLoader(test_set, **params, sampler=test_sampler)

    else:

        test_loader = torch.utils.data.DataLoader(test_set, **params)


    # Start predictions
    with torch.no_grad():

        # for j, (data, y, Mw,ava_stat, NOISE0, STF, index) in enumerate(test_loader):
        for j, (data_n, data , _, eq_par) in enumerate(test_loader):

            start = j*batch_size_per_gpu
            end = start + batch_size_per_gpu

            print ("j: {} -- Start {} - end: {}".format(j,start,end))


            # Load data and labels on GPU
            # data = data.to(gpu, non_blocking=True)
            data_n = data_n.to(gpu, non_blocking=True)

            recon_batch, mu, logvar = Dmodel(data_n.double())

            label = data.cpu().numpy()
            input_data = data_n.cpu()
            pred = recon_batch.cpu().numpy()

            # Rescale predictions
            scale = 1e-8
            recon_batch -= 1.0
            recon_batch *= scale


            TRUE[start:end,:,:,0] = label[:,0,:,:] # Label
            TRUE[start:end,:,:,1] = input_data[:,0,:,:] # Input data
            PREDS[start:end,:,:] = pred[:,0,:,:] #save reconstruction
            EQ_PAR[start:end, :] = eq_par[:,:3] # Save Mw lat and lon




    np.save(modelname+'_EQ_PAR_test_rank{}.npy'.format(PROCESS_RANK),EQ_PAR) 
    np.save(modelname+'_PREDS_test_rank{}.npy'.format(PROCESS_RANK),PREDS)
    np.save(modelname+'_TRUE_test_rank{}.npy'.format(PROCESS_RANK),TRUE)



def main(PROCESS_RANK, WORLD_SIZE, args):

    print("Running PEGSNET_DVAE (DDP) on rank {}.".format(PROCESS_RANK))

    if PROCESS_RANK==0:
        print("ME, PROCESS {}, I am MASTER".format(PROCESS_RANK))

    # Check which is the platform
    platform = args.platform
    # Set up distributed flag
    if platform == 'laptop':
        distributed = False
        LOCAL_RANK = 0

    elif platform == 'hotshot':
        distributed = True
        setup(PROCESS_RANK,WORLD_SIZE)
        LOCAL_RANK = PROCESS_RANK

    elif platform == 'JZ':
        distributed = True
        setup(PROCESS_RANK,WORLD_SIZE)
        LOCAL_RANK = idr_torch.local_rank


    set_backend('agg')  # for matplotlib Theo
    

    # Setting up the parameters
    # Dabatabse name with data
    working_db = args.db_name
    # Model output name
    modelprefix = args.model_outname
    ##################  Parameters of train/val/test sets ####################
    # Specifying these paraemters allows for smaller train/val/test for testing the algo
    # I have changed the way the index for train/val/test are used. Now they are fixed 
    # for each database and are loaded from ../DATABASES/run_db??_train_idx.npy
    train_nsamp = args.train_nsamp
    val_nsamp = args.val_nsamp
    test_nsamp = args.test_nsamp
    tot_samp = train_nsamp + val_nsamp + test_nsamp
    # Total batch size. This is divided by the number of GPUs
    batch_size = args.batch_size
    batch_size_per_gpu = batch_size // WORLD_SIZE
    # Learning rate
    lr = args.lr
    # Number of components to use
    ncomp = args.ncomp 
    # Number of epochs to run training for
    max_epochs = args.n_epochs
    # Set up an epoch start value. This will be incremented if training is resumed from checkpoint
    epoch_start = 1
    # Number of training steps for which we print infos
    log_interval = args.log_interval
    # Save a checkpoint every number of epochs 
    check_interval = args.check_interval
    # This is the path of an existing checkpoint from which we'll resume training
    resume_chkp = args.resume_chkp
    # Dimension of the bottleneck
    latent_dim = args.latent_dim
    # Which skip connections to establish
    skip_co = args.skip_co
    # Model to use for predictions on test set (BEST or LAST)
    test_mode = args.test_mode


    #Load index for train/val/test
    # Only select the firs n_samples based on the values specified as input 
    list_train = np.load(working_db +"_train_idx.npy")[:train_nsamp]
    list_val = np.load(working_db +"_val_idx.npy")[:val_nsamp]
    list_test = np.load(working_db +"_test_idx.npy")[:test_nsamp]

    if PROCESS_RANK ==  0:
        # Do some checks on dimensions and number of samples as specified in input parameters
        if (train_nsamp > len(list_train)):
            print (" train_nsamp is bigger than the maximum number of training samples for this database which is {}".format(len(list_train)))
            print (" Please reduce the value of train_nsamp")
            sys.exit("Exit")
        if (val_nsamp > len(list_val)):
            print (" val_nsamp is bigger than the maximum number of validation samples for this database which is {}".format(len(list_val)))
            print (" Please reduce the value of val_nsamp")
            sys.exit("Exit")
        if (test_nsamp > len(list_test)):
            print (" test_nsamp is bigger than the maximum number of testing samples for this database which is {}".format(len(list_test)))
            print (" Please reduce the value of test_nsamp")
            sys.exit("Exit")

    ################### CREATE OUTPUT DIRECTORY #########################################
    modeldir = "./outputs/model_" + modelprefix
    plotsdir = "./outputs/model_" + modelprefix + "/plots"


    modelname = modeldir + "/model_" + modelprefix + '_' + str(int(train_nsamp/1000)) + "k" + str(int(val_nsamp/1000)) + "k"
    plotsname = plotsdir + "/model_" + modelprefix + '_' + str(int(train_nsamp/1000)) + "k" + str(int(val_nsamp/1000)) + "k"

    if PROCESS_RANK ==0:
        if not os.path.exists(modeldir):
            os.makedirs(modeldir)
        if not os.path.exists(plotsdir):
            os.makedirs(plotsdir)

    # LOAD ACTIVE stations index and sort lon/lat 
    # USELESS now. The database has active stations already sorted
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


    # Data parameters
    t_max = 320
    n_stations = len(active_st_idx)-2
    n_comp = ncomp



    #################  CREATE MODEL ################
    if distributed:
        # Set the device
        torch.cuda.set_device(LOCAL_RANK)
        gpu = torch.device("cuda")
        # Create the model
        model0 = DVAE_WSC(n_comp, latent_dim, skips = skip_co).to(gpu)
        model0 = model0.double()
        # Wrap the model in a DDP object
        Dmodel = DDP(model0,device_ids=[LOCAL_RANK])
    else:
        gpu = torch.device("cuda")
        # Create the model
        model0 = DVAE_WSC(n_comp, latent_dim, skips = skip_co).to(gpu)
        Dmodel = model0.double().to(gpu)
        
    # Define the optimizer
    optimizer = optim.Adam(Dmodel.parameters(), lr=lr)


# IF RESUME load stuff and update epochs

    if resume_chkp is not None:
        # Path of the corresponding checkpoint
        CPATH = resume_chkp
        print ("")
        print(" Resuming training from {}".format(CPATH))
        
        if distributed:
            #Synchronize on all gpus
            dist.barrier()
            # By default the model is load to CPU. We need to remap to each GPU
            map_location = {'cuda:%d' % 0: 'cuda:%d' % LOCAL_RANK} # remap storage from GPU 0 to local GPU 
            # Now load it
            checkpoint = torch.load(CPATH, map_location=map_location)
            
        else:
            checkpoint = torch.load(CPATH, map_location=map_location)
            
        Dmodel.load_state_dict(checkpoint['model_state_dict']) # load checkpoint
        # Redefine stuff on CPU
        checkpoint = torch.load(CPATH)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch_start = checkpoint['epoch']
        # Update maxepoch
        max_epochs = epoch_start + max_epochs

#########################################################################
#############  SPECIFY training parameters and data loaders #############
    # Parameters
    if distributed:
        params = {'batch_size': batch_size_per_gpu,
                  'shuffle': False,
                  'num_workers': 40,
                  'pin_memory':True}
    else:
        params = {'batch_size': batch_size_per_gpu,
          'shuffle': True,
          'num_workers': 16,
          'pin_memory':True}
        
        
    # Datasets
    partition = {
        "train": list_train,
        "validation": list_val,
        "test" : list_test
        }# IDs


    # Generators
    training_set = HDF5Dataset(partition['train'],
                               n_stations,
                               t_max,
                               n_comp,
                               dshift = 0,
                               database_path=working_db+".hdf5"
                               )


    # Same for validation set. The only thing that changes is the sampler that takes care of uneven batches. 
    validation_set = HDF5Dataset(partition['validation'],
                                 n_stations,
                                 t_max,
                                 n_comp,
                                 dshift=0,
                                 database_path=working_db+".hdf5"
                                 )


    if distributed:
        #  The generator will load the data     

    # Sampler is needed by DistributedDataParallel
        train_sampler = torch.utils.data.distributed.DistributedSampler(training_set,
                                                                        num_replicas=WORLD_SIZE,
                                                                        rank=PROCESS_RANK)

    # The loader will load the data using generator and sampler
        train_loader = torch.utils.data.DataLoader(training_set, **params, sampler=train_sampler)


        val_sampler = DistributedEvalSampler(validation_set, num_replicas=WORLD_SIZE,
                                                                        rank=PROCESS_RANK)


        val_loader = torch.utils.data.DataLoader(validation_set, **params, sampler=val_sampler)

    else:

        train_loader = torch.utils.data.DataLoader(training_set, **params)
        val_loader = torch.utils.data.DataLoader(validation_set, **params)



########   START TRAINING LOOP 
    train_losses = []
    val_losses = []
    
######## coefficients for loss terms
    # BCE reconstruction loss
    alpha = 1   # default = 1
    # KLD latent loss
    beta  = 0.1 # default = 1     
    # SL smooth loss
    gamma = 0 # without smoothing
        
    BCE_train_losses, BCE_val_losses = [],[]
    KLD_train_losses, KLD_val_losses = [],[]
    SL_train_losses,  SL_val_losses  = [],[]
    
    if PROCESS_RANK == 0: start = datetime.now()
# This is looping over all epochs
    for epoch in range(epoch_start, max_epochs + 1):
        if PROCESS_RANK == 0:
            start_dataload = time()

##############   SAVE/LOAD CHECKPOINT 
        if epoch % check_interval == 0:

            CPATH = modelname + "_CHECK_EPOCH_{}.pth".format(epoch)

            # ONLY MASTER SAVES THE CHECKPOINT
            if PROCESS_RANK == 0:
                print ("+++ Saving checkpoint in {} +++".format(CPATH))
                #torch.save(Pmodel.state_dict(),CPATH)
                torch.save({
                            'epoch': epoch,
                            'model_state_dict': Dmodel.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                             }, CPATH)

            if distributed:
                # Set epoch is needed to avoid deterministic sampling 
                train_sampler.set_epoch(epoch)
    ################### NOW LOAD IT EVERYWHERE
                #Synchronize on all gpus
                dist.barrier()
                map_location = {'cuda:%d' % 0: 'cuda:%d' % LOCAL_RANK} # remap storage from GPU 0 to local GPU 
                #Pmodel.load_state_dict(torch.load(CPATH, map_location=map_location))
                checkpoint = torch.load(CPATH, map_location=map_location)
                Dmodel.load_state_dict(checkpoint['model_state_dict']) # load checkpoint
                #optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                #epoch = checkpoint['epoch']
            else:
                checkpoint = torch.load(CPATH)
                Dmodel.load_state_dict(checkpoint['model_state_dict']) # load checkpoint            

         # Set the model in train mode
        Dmodel.train()

        train_loss     = 0.0
        BCE_train_loss = 0.0
        KLD_train_loss = 0.0
        SL_train_loss  = 0.0
        
        # Start looping over batches for the training dataset
        for batch_idx, (data_n, data , _, _ ) in enumerate(train_loader):

            if PROCESS_RANK == 0: stop_dataload = time()

            # Load data and labels on GPU
            data = data.to(gpu, non_blocking=True)
            data_n = data_n.to(gpu, non_blocking=True)

            if PROCESS_RANK == 0: start_training = time()

            optimizer.zero_grad()

            # Predict labels using current batch
            recon_batch, mu, logvar = Dmodel(data_n.double())
            
            # Evaluate the loss for this batch
            BCE, KLD, SL = loss_function(recon_batch, data.view(-1,n_comp,t_max, n_stations),
                                         mu, logvar, 
                                         bceCoeff=alpha, kldCoeff=beta, slCoeff=gamma,
                                         recArg = args.loss)

            loss = BCE + KLD + SL
            
            #Loss is the mean over all examples in this batch. Default behaviour of criterion.
            train_loss     += loss.item()
            BCE_train_loss += BCE.item()
            KLD_train_loss += KLD.item()
            SL_train_loss  += SL.item()
            

            # Do backprop
            loss.backward()
            # Take a step of optimizer
            optimizer.step()


    #         print (" RANK: {} -- batchidx:{} -- \
    #                len(train_loader) {} -- \
    #                    len(dataset) {} -- len(index) {} \
    #                        -- loss {:.2f} -- train_loss {:.2f}".format(
    # PROCESS_RANK, batch_idx, 
    # len(train_loader), 
    # len(train_loader.dataset), 
    # len(index), loss.item(), train_loss))


            if PROCESS_RANK == 0: stop_training = time()

            # Print some infos after log_interval steps for this epoch
            if batch_idx % log_interval == 0 and PROCESS_RANK ==  0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\t BCE Loss: {:.6f}  KLD Loss: {:.6f}  SL Loss: {:.6f}  Time data load: {:.3f}ms, Time training: {:.3f}ms'.format(
                    epoch, batch_idx * len(data)*WORLD_SIZE, len(train_loader.dataset),
                    100. * batch_idx / len(train_loader),
                    BCE, KLD, SL, (stop_dataload - start_dataload)*1000, (stop_training - start_training)*1000 ))
                # print(len(data))
                if PROCESS_RANK == 0: start_dataload = time()

        nbatches = batch_idx+1
        # Save train loss for this epoch
        train_losses.append(train_loss/nbatches)
        BCE_train_losses.append(BCE_train_loss/nbatches)
        KLD_train_losses.append(KLD_train_loss/nbatches)
        SL_train_losses.append(SL_train_loss/nbatches)


        if PROCESS_RANK ==  0 or PROCESS_RANK==1:
            print('====> Epoch: {} Average loss: {:.4f} -- nbatches: {}'.format(
                  epoch, train_loss / nbatches, nbatches))


        if distributed:
            # Once the epoch is over check on the validation set   
            val_sampler.set_epoch(epoch)
            
        # Set model in eval mode but leave dropout active
        Dmodel.eval()
        
        test_loss    = 0.0
        BCE_val_loss = 0.0
        KLD_val_loss = 0.0
        SL_val_loss  = 0.0
        
        # No gradients are computed here.
        with torch.no_grad():
            if PROCESS_RANK ==0: val_start_dataload = time()

            # Start looping over batches of the validation set
            for i, (data_n, data, _,_) in enumerate(val_loader):
                if PROCESS_RANK == 0: val_stop_dataload = time()

                # Load data and labels on GPU
                data = data.to(gpu, non_blocking=True)
                data_n = data_n.to(gpu, non_blocking=True)

                if PROCESS_RANK == 0: start_testing = time()
                recon_batch, mu, logvar = Dmodel(data_n.double())

                tBCE, tKLD, tSL = loss_function(recon_batch, data.view(-1,n_comp,t_max, n_stations),
                                                mu, logvar,
                                                bceCoeff=alpha, kldCoeff=beta, slCoeff=gamma,
                                                recArg=args.loss)

                tloss = tBCE + tKLD + tSL
                
                test_loss    += tloss.item()
                BCE_val_loss += tBCE.item()
                KLD_val_loss += tKLD.item()
                SL_val_loss  += tSL.item()

                if PROCESS_RANK == 0: stop_testing = time()

                # Print some infos on the validation set 
                if i % log_interval == 0 and PROCESS_RANK ==  0:
                    print('Test Epoch: {} [{}/{} ({:.0f}%)]\t val loss: {:.6f} Time val data load: {:.3f}ms, Time testing: {:.3f}ms'.format(
                    epoch, i * len(data)*WORLD_SIZE, len(val_loader.dataset),
                    100. * i / len(val_loader),
                    tloss.item(), (val_stop_dataload - val_start_dataload)*1000, (stop_testing - start_testing)*1000 ))

                    if PROCESS_RANK == 0: val_start_dataload = time()

                if i == 0 and PROCESS_RANK==0:    # (first batch, rank 0)
                    n = min(data.size(0), 8)
                    comparison = torch.cat([data[:n,0].view(n,1,t_max,n_stations),
                                        data_n[:n,0].view(n,1,t_max,n_stations),
                                      recon_batch[:n,0].view(n,1,t_max,n_stations)])
                    if (epoch%10 == 0 or epoch==1):   # output image every 10 epochs only (and epoch 1)
                        save_image(comparison.cpu(), 
                                   plotsname + "_epoch" + str(epoch) + '_recon.png', 
                                   nrow=n, normalize=True, scale_each=True, padding=10)
                    
################### Plot comparison/acc using matplotlib+numpy: ###############################################
                        n = min(data.size(0),5)
                        # n_selec = np.arange(0,n,1) # select first n images in dataset
                        # n_selec = np.random.randint(0,data.size(0),size=n) # select randomly n images in dataset
                        
                        input_images  = data_n[:n,0].view(n,1,t_max,n_stations) #(N,C,H,W) 
                        target_images = data[:n,0].view(n,1,t_max,n_stations)
                        output_images = recon_batch[:n,0].view(n,1,t_max,n_stations)
                        
                        # plot acc_map and comp map (from visuals.py star import)
                        acc_map  = plot_multi_acc_map(output_images, target_images, clip=1, clip_bad=0.5, # 50% relative error = 0 in accuracy
                                                      fig_name= plotsname+"_epoch"+str(epoch)+"_accmap", path='')
                        
                        comp_map = plot_multi_comp_map(output_images, target_images,
                                                       fig_name= plotsname+"_epoch"+str(epoch)+"_compmap", path='')
                        
                        rm_noise = plot_multi_comp_map(input_images, output_images,
                                                       fig_name= plotsname+"_epoch"+str(epoch)+"_noisemap",path='')




        nbatches = i+1
        test_loss /= nbatches
        val_losses.append(test_loss)
        
        BCE_val_losses.append(BCE_val_loss/nbatches)
        KLD_val_losses.append(KLD_val_loss/nbatches)
        SL_val_losses.append(SL_val_loss/nbatches)



        if epoch == epoch_start:
            miniloss = test_loss
        if PROCESS_RANK ==  0 and epoch >  epoch_start:
            print('====> Test set loss: {:.4f} -- nbatches {}'.format(test_loss, nbatches))

            if test_loss < miniloss:
                print ('+++ Saving best model because test_loss: {} < miniloss: {} +++'.format(test_loss,miniloss))
                miniloss = test_loss
##############   SAVE/LOAD CHECKPOINT 
                PATH_BEST = modelname + "_BEST"
                if distributed:
                    # ONLY MASTER SAVES THE CHECKPOINT
                    torch.save(Dmodel.module.state_dict(), PATH_BEST+".pth")
                    torch.save(Dmodel.state_dict(), PATH_BEST+"_DDP.pth")
                else:
                    torch.save(Dmodel.state_dict(), PATH_BEST)
    if distributed:
        # Barrier at the end of each epoch. A waste but I want to be sure the procs are syncrhonized 
        dist.barrier()



#########################################################################
    
    ########### CHECK HOW LOSSES ARE SAVED ############
    # We might need to calculate exact losses as these are just 
    # mean losses for each batch. This can be biased if batches are of 
    # different sizes. 
    # Check Here:
    # https://discuss.pytorch.org/t/plotting-loss-curve/42632/3

    # For the moment save lossess for each rank. These will be averaged later
    np.savetxt(modelname +  "RANK_{}_train_losses.txt".format(PROCESS_RANK), np.array(train_losses))
    np.savetxt(modelname +  "RANK_{}_val_losses.txt".format(PROCESS_RANK), np.array(val_losses))
    
    BCE_loss = np.array( ['BCE', np.array(BCE_train_losses), np.array(BCE_val_losses) ] )
    KLD_loss = np.array( ['KLD', np.array(KLD_train_losses), np.array(KLD_val_losses) ] )
    SL_loss  = np.array( ['SL' , np.array(SL_train_losses),  np.array(SL_val_losses)  ] )
    all_sublosses = np.array([BCE_loss, KLD_loss, SL_loss])
    np.save(modelname +  "RANK_{}_all_sublosses.npy".format(PROCESS_RANK),all_sublosses)

    # MASTER NODE will save some stuff and copy some files in output folder
    if PROCESS_RANK==0:
        print(">>> Training complete in: " + str(datetime.now() - start))
        if distributed:
            torch.save(Dmodel.module.state_dict(), modelname +".pth")
            torch.save(Dmodel.state_dict(), modelname +"_DDP.pth")
        else:
            torch.save(Dmodel.state_dict(), modelname +".pth")
            
        #Save list of index for test set (any of these have been used in train+val)
        np.savetxt(modelname + "_test_idx.txt", list_test)
        print (modelname  + "_test_idx.txt has been saved in folder " + modeldir)

        np.savetxt(modelname  + "_val_idx.txt", list_val)
        print (modelname + "_val_idx.txt has been saved in folder " + modeldir)


        shutil.copy2(st_file, modelname +"_stat_input.txt")
        shutil.copy2("Classes.py", modelname +"_Classes.py")
        shutil.copy2("./run_script.sh", modelname +"_run_script.sh")
        
        
################### PLOT RANK_0 losses ######
        train_avg_gpu, val_avg_gpu = get_loss_arrays(modelname, path='', gpus = WORLD_SIZE)[:2]
        plot_loss_vs_epoch(train_avg_gpu, val_avg_gpu, 
                           fig_name= plotsname+'_gpu_loss_vs_epoch', path='')
        plot_different_losses(all_sublosses, 
                              fig_name= plotsname+'_allsublosses_vs_epoch', path='')
        

    if distributed:
        # BArrier at the end. Be sure that the files just copied are available to all ranks. 
        dist.barrier()

    if PROCESS_RANK==0:
        print("+++++++++++++++++++++++++++++++")
        print("Making predictions on test set")
        start_test_set = datetime.now()
    # Make predictions on test set 
    predict_test_set(Dmodel, modelname, working_db, (n_stations,t_max,ncomp), test_mode, distributed, WORLD_SIZE, LOCAL_RANK, PROCESS_RANK)

    if distributed:
        # BArrier at the end. Be sure that the files just copied are available to all ranks. 
        dist.barrier()


    if PROCESS_RANK==0:
        print(">>> Test set completed in: " + str(datetime.now() - start_test_set))

    if distributed:
        cleanup()

    if PROCESS_RANK==0:
        print("+++++++++++++++++++++++++++++++")
        print("Done")    



    


if __name__ == "__main__":
    
    # parse arguments
    args = parse_args()
    if args is None:
        print("No args found. Exit...")
        exit()
 
    
    
    if args.platform == 'hotshot':
        WORLD_SIZE = 8
        # Spawn mutliprocessing
        mp.spawn(main,
                  args=(WORLD_SIZE, args,),
                  nprocs=WORLD_SIZE,
                  join=True)
    else:
        # Define variables
        WORLD_SIZE = idr_torch.size
        RANK = idr_torch.rank

        main(RANK, WORLD_SIZE, args)



    


























