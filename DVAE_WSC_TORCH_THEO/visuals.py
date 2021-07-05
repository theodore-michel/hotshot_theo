# -*- coding: utf-8 -*-
"""
Created on Thu Jul  1 10:53:40 2021

@author: herod

Visuals python file 

For displaying denoiser outputs vs target and analyzing network performance.
More precisely:
    - Plot reconstruction comparison
    - Plot loss curves for training + validation (global loss)
    - Plot loss curves for training + validation (each loss: BCE, KLD, SL)
    - Plot accuracy maps
    
Call as import in run_train_model_torch.py 

"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import torch

### Read .txt files of losses for each rank for each epoch (mean over batch) and plot losses vs epochs (include margins)

def get_loss_arrays(filename_prefix,path,gpus=4):
    '''takes in .txt filename prefix (modelname) and returns train/val losses
    averaged over all gpus, as well as arrays of train and validation
    losses in format shape=(N_epochs,#_of_gpus)'''
                                                                   
    suffix_train = "_train_losses.txt"
    suffix_val   = "_val_losses.txt"
    train_losses = []
    val_losses   = []
    
    for rank in range(gpus):
        filename = filename_prefix+"_RANK"+str(rank)
        train_losses.append(np.loadtxt(path+filename+suffix_train))
        val_losses.append(np.loadtxt(path+filename+suffix_val))
        
    train_losses_np = np.swapaxes(train_losses,0,1)
    val_losses_np   = np.swapaxes(val_losses,0,1)
    
    train_mean_gpus = np.mean(train_losses_np, axis=1)
    val_mean_gpus   = np.mean(val_losses_np, axis=1)
    
    return(train_mean_gpus,val_mean_gpus, train_losses_np, val_losses_np)


def get_array(filename,path):
    return(np.loadtxt(path+filename))


def plot_loss_vs_epoch(train_loss,val_loss,fig_name='loss_vs_epoch_fig',path=''):
    '''Plots train and validation loss vs epochs. Note that this is only for 
    1D arrays of train losses and validation losses, so mean over all gpus.
    
    Saves the figure as png to specified path (default is current working
    directory) with given fig_name.'''
    
    epochs = np.arange(len(train_loss))
    figure, plot = plt.subplots(1,1)
    
    plot.plot(epochs, train_loss, label = "train")
    plot.plot(epochs, val_loss,'-.', label = "validation")
    
    plot.set_xlabel('Epochs')
    plot.set_ylabel('Loss (avg over GPUs)')
    plot.tick_params(labelright=True, right=True)
    
    plot.legend(loc="upper right")
    plot.set_title("Average loss over all GPUs vs epochs")
    
    #plt.show()
    
    # save plot as png
    plt.savefig(path+fig_name,format='png',bbox_inches='tight')
    
    #return(path+fig_name)


def plot_multiloss_vs_epoch(train_losses,val_losses, legend_labels,fig_name='losses_vs_epoch_fig',path=''): 
    '''Plots train and validation losses vs epochs. Note that this is adapted
    to 2D arrays of losses so multiple GPU losses or different kinds of losses 
    (BCE,KLD,SL,...). All plots are on the same figure with different colors. Legend 
    labels must be given to keep track of plot lines, will return error if not correct size.
    
    Order of legend labels are same as order in arrays second dimension. Plot is saved as png
    to specified path (default is current working directory) with given fig_name''' 
    
    epochs = np.arange(train_losses.shape[0])
    numPlt = train_losses.shape[1]
    
    if numPlt != len(legend_labels):
        print(str(numPlt)+" curves to plot but {} legend labels given".format(len(legend_labels)))
        exit()
    
    colors = cm.rainbow(np.linspace(0,1,numPlt))
    figure, plot = plt.subplots(1,1)
    for i in range(numPlt):
        plot.plot(epochs, train_losses[i],    c=colors[i], label = "train " + legend_labels[i])
        plot.plot(epochs, val_losses[i],'-.', c=colors[i], label = "validation " + legend_labels[i])
    
    plot.set_xlabel('Epochs')
    plot.set_ylabel('Losses')
    plot.tick_params(labelright=True, right=True)
    
    plot.legend(loc="upper right")
    plot.set_title("Average losses over all batches vs Epochs")
    
    #plt.show()
    
    # save plot as png
    plt.savefig(path+fig_name,format='png',bbox_inches='tight')
    
    #return(path+fig_name)
    
    
    

### Take in tensors (output and target) and construct accuracy map, save as png

def NCHW_to_NHW_np(tensor):
    '''Converts NCHW tensor to numpy NHW array'''
    tensor_numpy = tensor.cpu().detach().numpy()
    tensor_numpy = np.transpose(tensor_numpy, axes=(0,3,1,2)).squeeze()
    return(tensor_numpy)

#  ### OLD ACCURACY MAP VERSION
# def diff_map(output,target, clip=1):
#     '''maps absolute difference between output and target,
#     output and target are 2D (H,W) tensors or (H,W,1) tensors.'''
#     diff = torch.clamp(torch.abs(target-output), min=0, max=clip)
#     return(diff)

# def acc_map_torch(output,target,clip=1,residual=1):
#     '''maps accuracy in percentage between output and target,
#     output and target are 2D (H,W) tensors or (H,W,1) tensors.'''
#     diff = diff_map(output,target,clip=clip)
#     acc  = torch.div(diff,torch.abs(target))
#     acc  = torch.clamp(acc,min=0, max=residual)
#     acc *= (-1/residual) # normalise to -1 -> 0 with clipping at value=residual for worst results
#     acc += 1    # note that acc is in 0-1 range, 1 corresponds to perfect accuracy, 
#                 # 0 is arbitrary bad acc (if residual=1, corresponds to 100% relative error)
#     return(acc)

# def plot_acc_map(accuracy_map,fig_name='accuracy_map_fig',path=''):
#     if len(accuracy_map.shape)>2:
#         if accuracy_map.shape[2]==1:
#             accuracy_map = accuracy_map.reshape(accuracy_map.shape[0],accuracy_map.shape[1])
#         else: 
#             print("Wrong shape for accuracy map: (H,W,1) or (H,W)")
#             exit()
#     acc_numpy = accuracy_map
    
#     figure, (cbar,plot) = plt.subplots(2,1,     # colorbar will be on top
#                               gridspec_kw={"height_ratios":[0.05,1]}) # cbar_size = 1/20 of im_size
#     im = plot.imshow(acc_numpy,cmap='viridis')
#     plot.set_xlabel("Stations")
#     cb = figure.colorbar(im,cax=cbar, orientation="horizontal", label="accuracy")
#     cb.ax.xaxis.set_label_position('top')
#     plt.suptitle('Accuracy Map of Denoiser Output')
#     # plt.show()
    
#     plt.savefig(path+fig_name,format='png',bbox_inches='tight')
    
#     # return(path+fig_name)
    

def acc_map(output, target, clip=1, residual=1):
    '''output and target are NHW numpy arrays, this function computes the accuracy map and
    returns it in format NHW.'''
    diff = np.clip( np.abs( output-target ) ,0 , clip)
    acc  = np.divide(diff,np.abs(target))
    acc  = np.clip(acc,0,residual)
    acc *= (-1/residual)    # normalise to -1 -> 0 with clipping at value=residual for worst results
    acc += 1    # note that acc is in 0-1 range, 1 corresponds to perfect accuracy, 
                # 0 is arbitrary bad acc (if residual=1, corresponds to 100% relative error)
    return(acc)
    
def plot_multi_acc_map(output, target, clip=1, residual=1, fig_name="accuracy_map_fig", path=''):
    '''returns acc_map of shape NHW'''
    n = output.shape[0]
    out_np  = NCHW_to_NHW_np(output)
    targ_np = NCHW_to_NHW_np(target)
    acc_np  = acc_map(out_np, targ_np, clip=clip, residual=residual)
    
    nrow = int(np.sqrt(n))
    ncol = n//nrow + 1
    fig, axes = plt.subplots(nrows=nrow, ncols=ncol)
    for ax,index in zip(axes.flat,range(n)):
        im = ax.imshow(acc_np[index], vmin=0, vmax=1)
        ax.set_xlabel("Stations")
        
    # fig.subplots_adjust(right=0.8)
    # cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    # fig.colorbar(im, cax=cbar_ax)
    fig.colorbar(im,ax=axes.ravel().tolist(),label="accuracy")       
    # source: https://stackoverflow.com/questions/13784201/matplotlib-2-subplots-1-colorbar
    plt.suptitle('Accuracy map of denoiser output')
    # plt.show()
    
    ### map subplots another way: n colums, 3 or 4 rows (targ, out, acc, (in))
    
    plt.savefig(path+fig_name,format='png',bbox_inches='tight')
    
    return(acc_np)
    

    

### Take in tensors (output, target, input) and construct comparison map (scramble): input, target, output-target diff. show extracted noise?



### Take in specific losses (BCE, KLD, SL) and plot for each epoch, include relative importance in global loss (with coeffs alpha beta gamma)



