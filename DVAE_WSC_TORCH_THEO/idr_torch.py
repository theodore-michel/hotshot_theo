#!/usr/bin/env python
# coding: utf-8

import os

###### FOR JEAN-ZAY
import hostlist


################# Set up variables #################
###### FOR JEAN-ZAY
# # get SLURM variables
rank = int(os.environ['SLURM_PROCID'])
local_rank = int(os.environ['SLURM_LOCALID'])
size = int(os.environ['SLURM_NTASKS'])
cpus_per_task = int(os.environ['SLURM_CPUS_PER_TASK'])




###### FOR LAPTOP
#rank = 0
#size = 1
#cpus_per_task = 8

#################  get node list from slurm ################# 
###### FOR JEAN-ZAY
hostnames = hostlist.expand_hostlist(os.environ['SLURM_JOB_NODELIST'])

#################  get IDs of reserved GPU ################# 
###### FOR JEAN-ZAY
gpu_ids = os.environ['SLURM_STEP_GPUS'].split(",")

###### FOR HOTSHOT
#gpu_ids = [0, 1, 2, 3, 4, 5, 6, 7]

###### FOR LAPTOP
# gpu_ids = 0 

################# define MASTER_ADD & MASTER_PORT #################
######## FOR JEAN-ZAY
os.environ['MASTER_ADDR'] = hostnames[0]
os.environ['MASTER_PORT'] = str(12345 + int(min(gpu_ids))) # to avoid port conflict on the same node

######## FOR HOTSHOT 
#os.environ['MASTER_ADDR'] = 'localhost'
#os.environ['MASTER_PORT'] = '12355'
