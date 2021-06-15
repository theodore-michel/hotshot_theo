#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 17:49:02 2019

@author: quentin

@modified: andrea 15/01/2019 to treat pegs as a three components object
"""

import numpy as np
import math
from scipy.interpolate import griddata
from scipy import signal
#from scipy.integrate import cumtrapz
import STFlib

from obspy.signal.rotate import rotate_rt_ne
from obspy.signal.rotate import rotate2zne
from obspy.geodetics.base import gps2dist_azimuth

def get_pegs(event_name, moment_rate, sos1, sos2, depth, filte=True):
    
    
#   Modify this to point to your root folder
#    homedir='/home/licciar/Andrea_hotshot/'
    homedir='/home/licciar/Andrea_generate_database_v2.0/'
#    homedir='/home/andrea/PEGS/Generate_database_v2.0/'

    pegs_dir_file = str(homedir) + '/pegs_npz/PEGS.dir.40km.' + event_name + '.' + str(int(depth)) + 'KM.npz'
    pegs_ind_file = str(homedir) + '/pegs_npz/PEGS.ind.40km.' + event_name + '.' + str(int(depth)) + 'KM.npz'
    
    pegs_dir = np.load(pegs_dir_file)
    pegs_ind = np.load(pegs_ind_file)
    
    lats = pegs_dir['latitude.0']
    lons = pegs_dir['longitude.0']
    pwav = pegs_dir['tp']
    
    # Now reads all three components which are stored in a single matrix of dim(len(pwav),350,3)
    # pegs[:,:,0] = z, vertical
    # pegs[:,:,1] = r, radial
    # pegs[:,:,2] = t, transverse 
    pegsz = pegs_dir['raw.z'][:, :350] + pegs_ind['raw.z'][:, :350]
    
    pegs = np.zeros((pegsz.shape[0],pegsz.shape[1],3))
    pegs[:,:,0] = pegsz
    pegs[:,:,1] = pegs_dir['raw.r'][:, :350] + pegs_ind['raw.r'][:, :350]
    pegs[:,:,2] = pegs_dir['raw.t'][:, :350] + pegs_ind['raw.t'][:, :350]
    
    for idx in range(3):
        for i, peg in enumerate(pegs[:,:,idx]):
            pegs[i,:,idx] = np.convolve(peg, moment_rate)[:len(peg)]
        
        if filte:
            pegs[:,:,idx] = signal.sosfilt(sos1, pegs[:,:,idx])
            pegs[:,:,idx] = signal.sosfilt(sos2, pegs[:,:,idx])
    
    return lons, lats, pwav, pegs



    

def synthetic_generator(Mw,longitude, latitude, depth,dip, strike, rake,st_lon, st_lat, sos1, sos2, filt=True):
    
    
   # Modified by AL 16/01/2020:
   # 0) Input modified to have a matrix of EQ_PARAMS
   # 1) Now it gets the depth (20 or 30 km) of source from the list of earthquake parameters from Quentin
   # 2) Now it generates the STF randomly using STFlib.compute_STF located in STFlib.py
   # 3) Now it outputs three components (RTZ) pegs
   
   #   INPUTS:
   #           -->  EQ_PARAMS is a database of earthquake parameters (n_eq, 5)
   #                EQ_PARAMS[:,0] = LONGITUDE
   #                EQ_PARAMS[:,1] = LATITUDE
   #                EQ_PARAMS[:,2] = DEPTH
   #                EQ_PARAMS[:,3] = DIP
   #                EQ_PARAMS[:,4] = STRIKE
   #           --> st_lon(nstat,1) stations longitude
   #           --> st_lat(nstat,2) stations latitude
   #           --> sos1, sos2 are sos objects for filtering defined in the main script 
   
    deg2rad = np.pi / 180.0
    rad2deg = 180.0 / np.pi
    rt = 6371000.0
      
    # Randomly pick Mw ~U[7.0,10.0]
#    Mw = np.random.uniform(6.5,10.0)         
    # Calculate M0
    moment0 = math.pow(10,1.5*Mw+9.1)

#    longitude = EQ_PARAMS[k,0]
#    latitude = EQ_PARAMS[k,1]
#    depth = EQ_PARAMS[k,2]
#    dip = EQ_PARAMS[k,3]
#    strike = EQ_PARAMS[k,4]

    # Randomly pick rake ~N(90,10)
#    rake = np.random.normal(90,10)
#    rake = 95.0
    
    # Randomly get a moment rate (STF)
#    moment_rate = STFlib.compute_STF(Mw,noise=True)
#    Mw=8.5
    
    moment_rate = STFlib.compute_STF(Mw,noise=True)
    
    
    lon1, lat1, pwav, pegs1 = get_pegs('MTENSOR1', moment_rate, sos1, sos2, depth, filte=filt)
    lon1, lat1, pwav, pegs2 = get_pegs('MTENSOR2', moment_rate, sos1, sos2, depth, filte=filt)
    lon1, lat1, pwav, pegs3 = get_pegs('MTENSOR3', moment_rate, sos1, sos2, depth, filte=filt)
    lon1, lat1, pwav, pegs4 = get_pegs('MTENSOR4', moment_rate, sos1, sos2, depth, filte=filt)
    
    a1 = + np.cos(1.*dip * deg2rad) * np.cos(rake * deg2rad) * moment0 / 1.0e22
    a2 = + np.sin(1.*dip * deg2rad) * np.cos(rake * deg2rad) * moment0 / 1.0e22
    a3 = - np.cos(2.*dip * deg2rad) * np.sin(rake * deg2rad) * moment0 / 1.0e22
    a4 = + np.sin(2.*dip * deg2rad) * np.sin(rake * deg2rad) * moment0 / 1.0e22
    
    pegs = a1*pegs1 + a2*pegs2 + a3*pegs3 + a4*pegs4
    
    phi1 = latitude*deg2rad
    lambda1 = longitude*deg2rad
    theta0 = np.arctan2( np.sin(lon1*deg2rad)*np.cos(lat1*deg2rad), np.sin(lat1*deg2rad) )
    theta = theta0 + strike*deg2rad

    a = np.sin(lat1*deg2rad/2)*np.sin(lat1*deg2rad/2) + np.cos(lat1*deg2rad) * np.sin(lon1*deg2rad/2)*np.sin(lon1*deg2rad/2)
    delta = 2 * np.arctan2( np.sqrt(a), np.sqrt(1-a) )

    phi2 = np.arcsin( np.sin(phi1) * np.cos(delta) + np.cos(phi1) * np.sin(delta) * np.cos(theta) )
    lambda2 = lambda1 + np.arctan2( np.sin(theta)*np.sin(delta)*np.cos(phi1), np.cos(delta)-np.sin(phi1)*np.sin(phi2) )    

    lonr = lambda2 * rad2deg
    latr = phi2 * rad2deg
     
    pwav_st = griddata((lonr, latr), pwav, (st_lon, st_lat), method='linear') 
    
    # create output array for pegs at station location
    pegs_st = np.zeros((len(st_lon), 350 ,3))
    pegs_st[:,:,0] = griddata((lonr, latr), pegs[:,:,0], (st_lon, st_lat), method='linear')
    pegs_st[:,:,1] = griddata((lonr, latr), pegs[:,:,1], (st_lon, st_lat), method='linear')
    pegs_st[:,:,2] = griddata((lonr, latr), pegs[:,:,2], (st_lon, st_lat), method='linear')
    
    
    

# Rotate from RT to NE    
    flag = True
#    flag= False
    pegs_st2 = np.zeros((len(st_lon), 350 ,3))
    pegs_st2[:,:,0] = pegs_st[:,:,0]
    if flag == True:
        for i in range(len(st_lon)):
            # Get backazimuth
             DIST,BAZ, BAZ2 = gps2dist_azimuth(st_lat[i],st_lon[i],latitude,longitude)
#             print(st_lat[i],st_lon[i], latitude,longitude, DIST, BAZ, BAZ2)
             
             # Rotate RT to NE
#             if (BAZ>180.0):
#                 angle = BAZ - 180.0
#             else:
#                 angle = BAZ + 180
##                 
#                 
#             pegs_st2[i,:,1],pegs_st2[i,:,2] = rotate_horizontal(pegs_st[i,:,1],pegs_st[i,:,2], angle)

             pegs_st2[i,:,1],pegs_st2[i,:,2] = rotate_rt_ne(pegs_st[i,:,1],pegs_st[i,:,2],BAZ)


    
    return pwav_st, pegs_st2

































