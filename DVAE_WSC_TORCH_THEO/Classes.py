#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  9 11:17:46 2020

@author: licciar
"""


import torch
import h5py
import numpy as np
from torch.utils.data import Sampler
import torch.distributed as dist

from torch import nn
from torch.nn import functional as F
import numpy as np

class DistributedEvalSampler(Sampler):

    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=False, seed=0):
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        # self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.num_replicas))
        # self.total_size = self.num_samples * self.num_replicas
        self.total_size = len(self.dataset)         # true value without extra samples
        indices = list(range(self.total_size))
        indices = indices[self.rank:self.total_size:self.num_replicas]
        self.num_samples = len(indices)             # true value without extra samples

        self.shuffle = shuffle
        self.seed = seed

    def __iter__(self):
        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = list(range(len(self.dataset)))


        # # add extra samples to make it evenly divisible
        # indices += indices[:(self.total_size - len(indices))]
        # assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        r"""
        Sets the epoch for this sampler. When :attr:`shuffle=True`, this ensures all replicas
        use a different random ordering for each epoch. Otherwise, the next iteration of this
        sampler will yield the same ordering.
        Arguments:
            epoch (int): _epoch number.
        """
        self.epoch = epoch

class DVAE(nn.Module):
    def __init__(self, n_chan,latent_dim):
        super(DVAE, self).__init__()
        
        
        self.n_chan = n_chan
        self.height = 320
        self.width =  72
        self.h_dim  = 8 
        self.latent_dim = latent_dim
        self.enc = nn.Sequential(
                                    nn.Conv2d(self.n_chan, self.h_dim*2, 3,2,1),
                                    nn.LeakyReLU(0.2),

                                    
                                    nn.Conv2d(self.h_dim*2,self.h_dim*4, 3,2,1),
                                    nn.BatchNorm2d(self.h_dim*4),
                                    nn.LeakyReLU(0.2),

                                    
                                    nn.Conv2d(self.h_dim*4,self.h_dim*8,3,2,1),
                                    nn.BatchNorm2d(self.h_dim*8),
                                    nn.LeakyReLU(0.2)


                                    )
        
        indim = self.h_dim*8 * (self.height//8) * (self.width//8)
        self.z_mean = nn.Linear(indim,self.latent_dim)
        self.z_log_var = nn.Linear(indim,self.latent_dim)
        
        
        self.dec1 = nn.Sequential(nn.Linear(self.latent_dim, indim),
                                  nn.BatchNorm1d(indim),
                                  nn.ReLU()
                                  )
        self.dec2 = nn.Sequential(nn.ConvTranspose2d(self.h_dim*8, self.h_dim*4, 2, 2),
                                  nn.BatchNorm2d(self.h_dim*4),
                                  nn.ReLU(),
                                  
                                  nn.ConvTranspose2d(self.h_dim*4, self.h_dim*2 ,2, 2),
                                  nn.BatchNorm2d(self.h_dim*2),
                                  nn.ReLU(),
                                  
                                  nn.ConvTranspose2d(self.h_dim*2, self.n_chan, 2,2),
                                  nn.Sigmoid()
                                    )
        
        # self.t_conv4 = nn.Sequential(
        #                             nn.ConvTranspose2d(h_dim*2, self.n_chan, 2, stride=2),
        #                             nn.Sigmoid()
        #                             )
        


    def encode(self, e):
        e = self.enc(e)
        
        input_dim = self.h_dim*8 * self.height//8 * self.width//8
        
        h1 = e.view(-1,input_dim)
        
        return self.z_mean(h1), self.z_log_var(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, inp):
        
        # input_dim = self.latent_dim*8 * self.height//8 * self.width//8
        d_input = inp.view(-1,self.h_dim*8, self.height//8 , self.width//8)
        h3 = self.dec2(d_input)

        
        return h3

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        z = self.dec1(z)
        z = self.decode(z)
        
        return z.view(-1,self.n_chan, self.height,self.width), mu, logvar


class DVAE_WSC(nn.Module):
    # DVAE model with skip connections between Conv and ConvTranspose layer 1,2,3
    def __init__(self, n_chan,latent_dim,skips='111'): # [skip conv1/convT3, skip conv2/convT2, skip conv3/convT1]
                                                       # format is '111', 1 for true 0 for false
        super(DVAE_WSC, self).__init__()
        
        
        self.n_chan = n_chan
        self.height = 320
        self.width =  72
        self.h_dim  = 8 
        self.latent_dim = latent_dim
        self.skips = skips # do we do the skips?
        
        
        self.conv1 = nn.Sequential(
                                    nn.Conv2d(self.n_chan, self.h_dim*2, 3,2,1),
                                    nn.BatchNorm2d(self.hdim*2), # added batchnorm so output is similar to deconv3 input
                                    nn.LeakyReLU(0.2)
                                    )
        self.conv2 = nn.Sequential(
                                    nn.Conv2d(self.h_dim*2,self.h_dim*4, 3,2,1),
                                    nn.BatchNorm2d(self.h_dim*4),
                                    nn.LeakyReLU(0.2)
                                    )
        self.conv3 = nn.Sequential(
                                    nn.Conv2d(self.h_dim*4,self.h_dim*8,3,2,1),
                                    nn.BatchNorm2d(self.h_dim*8),
                                    nn.LeakyReLU(0.2)
                                    )
        
        
        indim = self.h_dim*8 * (self.height//8) * (self.width//8)
        self.z_mean = nn.Linear(indim,self.latent_dim)
        self.z_log_var = nn.Linear(indim,self.latent_dim)
        
        
        self.dec1 = nn.Sequential(nn.Linear(self.latent_dim, indim),
                                  nn.BatchNorm1d(indim),
                                  nn.ReLU()
                                  )
        
        
        self.deconv1 = nn.Sequential(   # will be receiving skip connection of conv3
                                     nn.ConvTranspose2d(self.h_dim*8, self.h_dim*4, 2, 2),
                                     nn.BatchNorm2d(self.h_dim*4),
                                     nn.ReLU()
                                     )
        self.deconv2 = nn.Sequential( # will be receiving skip connection of conv2
                                     nn.ConvTranspose2d(self.h_dim*4, self.h_dim*2 ,2, 2),
                                     nn.BatchNorm2d(self.h_dim*2),
                                     nn.ReLU()
                                     )
        self.deconv3 = nn.Sequetial(
                                    nn.ConvTranspose2d(self.h_dim*2, self.n_chan, 2,2),
                                    nn.Sigmoid()
                                    )
        
        
        self.skip = nn.Identity()


    def codings(self,e):
        # e is output of conv3
        input_dim = self.h_dim*8 * self.height//8 * self.width//8
        h1 = e.view(-1,input_dim)
        return self.z_mean(h1), self.z_log_var(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def prep_deconv(self, inp):
        # inp is output of dec1
        d_input = inp.view(-1,self.h_dim*8, self.height//8 , self.width//8)
        return d_input


    def forward(self, x):

        # encoder
        out = self.conv1(x)
        for_skip3 = self.skip(out)
        out = self.conv2(out)
        for_skip2 = self.skip(out)
        out = self.conv3(out)
        for_skip1 = self.skip(out)
        
        # codings
        mu, logvar = self.codings(out)
        out = self.reparameterize(mu,logvar)
        out = self.dec1(out)
        out = self.prep_deconv(out)
        
        #decoder
        if bool(self.skips[2]) : out += for_skip1     # output of conv3 is added to input of deconv1
        out = self.deconv1(out)
        if bool(self.skips[1]) : out += for_skip2     # output of conv2 is added to input of deconv2
        out = self.deconv2(out)
        if bool(self.skips[0]) : out += for_skip3     # output of conv1 is added to input of deconv3
        out = self.deconv3(out)
        
        
        
        return out.view(-1,self.n_chan, self.height,self.width), mu, logvar


class HDF5Dataset(torch.utils.data.Dataset):

  'Characterizes a dataset for PyTorch'
  def __init__(self, list_IDs, n_stat, t_max, n_comp,database_path="../DATABASES/run_db13.hdf5"):
        'Initialization'
        super(HDF5Dataset, self).__init__()
        
        self.n_stat = n_stat
        self.t_max = t_max
        self.n_comp = n_comp
        self.database_path = database_path
        self.list_IDs = list_IDs

  def open_hdf5(self):
        self.file = h5py.File(self.database_path, 'r')


  def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

  def __getitem__(self, index):
        'Generates one sample of data'
        if not hasattr(self, self.database_path):
            self.open_hdf5()

        # Select sample
        ID = self.list_IDs[index]
        
        # Get a time shift at random between 0 and tmax
        shift = np.random.randint(0,self.t_max+1)
        shift = 0   
     
        # starting sample
        t1 = 350-shift
        # ending sample
        t2 = t1+self.t_max
        # samples with respect to origin
        t3 = self.t_max - shift

        
        # Get the relevant window
        X = np.array(self.file['data'][ID,:self.n_stat,t1:t2,:self.n_comp ])
        # get pwave arrivals
        pwav = np.array(self.file['ptime'][ID,:self.n_stat ])
        label = np.array(self.file['label'][ID, :self.n_stat,t1:t2,:self.n_comp ]) 
        eq_params= np.array(self.file['eq_params'][ID,: ])

        
        ## Find muted/no-noise-stations
        idd = np.where(~X[:,:,0].any(axis=1))[0]   
        ## Mute labels as well. It should be the case already
        label[idd,:,:] = 0.0
        
########################################################################################        
## SET UP AMPLITUDES
# Also set label amplitudes to zero for muted stations
# P-Wave arrival
        # indP = np.array(np.floor(pwav), dtype=int)
        # # For each station all components
        # for i in range(X.shape[0]):
            
        #     # if a trace is all zeros this trace 
        #     # has been muted or there is not data noise for it.
        #     # Therefore, set its label to zero
        #     if not np.count_nonzero(X[i,:,:]):
        #         label[i,:,:] = 0.0
                
            
            
        #     # Set amplitudes to zero after P-wave arrival
        #     X[i, indP[i]:,:] = 0.0 
        #     label[i, indP[i]:,:] = 0.0
              
        X = np.nan_to_num(X)
        label = np.nan_to_num(label)  
        scale = 1e-8
        X = np.clip(X,-1.0*scale, scale)
        X /= scale
        X+=1.0
        label /= scale
        label+=1.0


        # Got to swap axes because pytorch has channel first
        X = np.swapaxes(X,-1,0)
        
        label = np.swapaxes(label,-1,0)
            
            
        # Load data and get label
        x = torch.from_numpy(X)
        y = torch.from_numpy(label)
        
        return x, y, pwav, eq_params 
    
    
        
    
