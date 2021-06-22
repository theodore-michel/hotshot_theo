#!/usr/bin/env python
# coding: utf-8
 
import os
# import hostlist
 

rank = 0
# local_rank = int(os.environ['SLURM_LOCALID'])
# size = int(os.environ['SLURM_NTASKS'])
size = 1
cpus_per_task = 8 #int(os.environ['SLURM_CPUS_PER_TASK'])
 
# get node list from slurm
# hostnames = hostlist.expand_hostlist(os.environ['SLURM_JOB_NODELIST'])
 
# get IDs of reserved GPU
gpu_ids = 0 #os.environ['SLURM_STEP_GPUS'].split(",")
 
# define MASTER_ADD & MASTER_PORT
# os.environ['MASTER_ADDR'] = hostnames[0]
# os.environ['MASTER_PORT'] = str(12345 + int(min(gpu_ids))) # to avoid port conflict on the same node