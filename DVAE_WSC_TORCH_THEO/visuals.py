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
import os

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
        filename = filename_prefix+"RANK_"+str(rank)
        train_losses.append(np.loadtxt(path+filename+suffix_train))
        val_losses.append(np.loadtxt(path+filename+suffix_val))
        
    train_losses_np = np.swapaxes(train_losses,0,1)
    val_losses_np   = np.swapaxes(val_losses,0,1)
    
    train_mean_gpus = np.mean(train_losses_np, axis=1)
    val_mean_gpus   = np.mean(val_losses_np, axis=1)
    
    return(train_mean_gpus,val_mean_gpus, train_losses_np, val_losses_np)


def get_array(filename,path):
    '''loads .txt or .npy files into array, note that extension is included in filename'''
    if filename[-4:] == '.txt' : array = np.loadtxt(path+filename) 
    if filename[-4:] == '.npy' : array = np.load(path+filename)
    else : 
        print("non-supported file format")
        exit()
    return(array)


def plot_loss_vs_epoch(train_loss,val_loss,fig_name='loss_vs_epoch_fig',path=os.getcwd()):
    '''Plots train and validation loss vs epochs. Note that this is only for 
    1D arrays of train losses and validation losses, so mean over all gpus.
    
    Saves the figure as png to specified path (default is current working
    directory) with given fig_name.'''
    
    epochs = np.arange(len(train_loss))
    # erase first 5 epochs since loss descales the plot
    # erase = min(train_loss.shape[0],5)-1
    
    figure, plot = plt.subplots(1,1)
    figure.set_size_inches((12.0, 12.0), forward=False)
    
    plot.plot(epochs, train_loss, label = "train")
    plot.plot(epochs, val_loss,'-.', label = "validation")
    
    plot.set_xlabel('Epochs')
    plot.set_ylabel('Loss (avg over GPUs)')
    up_bound_95 = max(np.quantile(train_loss,q=0.85), 
                      np.quantile(val_loss,q=0.85))
    low_bound   = 0.85 * min(train_loss.min(), val_loss.min())
    plot.set_ylim(bottom=low_bound,
                  top=up_bound_95)
    # plot.set_ylim(top=min(train_loss[erase],val_loss[erase]),
    #               bottom = 0.9*min(min(train_loss),min(val_loss)))
    plot.tick_params(labelright=True, right=True)
    
    plot.legend(loc="upper right")
    plot.set_title("Average loss over all GPUs vs epochs")
    
    figure.tight_layout()
    # plt.show()
    
    # save plot as png
    figure.savefig(path+fig_name+'.png',bbox_inches='tight',dpi=100)
    plt.close(figure)
    
    #return(path+fig_name)


def plot_multiloss_vs_epoch(train_losses,val_losses, legend_labels,fig_name='losses_vs_epoch_fig',path=os.getcwd()): 
    '''Plots train and validation losses vs epochs. Note that this is adapted
    to 2D arrays of losses so multiple GPU losses or different kinds of losses 
    (BCE,KLD,SL,...). All plots are on the same figure with different colors. Legend 
    labels must be given to keep track of plot lines, will return error if not correct size.
    
    Order of legend labels are same as order in arrays second dimension. Plot is saved as png
    to specified path (default is current working directory) with given fig_name''' 
    
    epochs = np.arange(train_losses.shape[0])
    numPlt = train_losses.shape[1]
    # erase = min(epochs.shape[0],5)-1
    
    if numPlt != len(legend_labels):
        print("{} curves to plot but {} legend labels given".format(numPlt,len(legend_labels)))
        exit()
    
    colors = cm.rainbow(np.linspace(0,1,numPlt))
    figure, plot = plt.subplots(1,1)
    figure.set_size_inches((12.0, 12.0), forward=False)
    for i in range(numPlt):
        plot.plot(epochs, train_losses[:,i],    c=colors[i], label = "train " + legend_labels[i])
        plot.plot(epochs, val_losses[:,i],'-.', c=colors[i], label = "val " + legend_labels[i])
        # plot.set_ylim(top=min(val_losses[erase,i],train_losses[erase,i]), 
        #               bottom = 0.9*min(min(train_losses[:,i]),min(val_losses[:,i])))
        up_bound_95 = max(np.quantile(train_losses[:,i],q=0.85), 
                          np.quantile(val_losses[:,i],q=0.85))
        low_bound   = 0.85 * min(train_losses[:,i].min(), val_losses[:,i].min())
        plot.set_ylim(bottom=low_bound,
                      top=up_bound_95)
    
    plot.set_xlabel('Epochs')
    plot.set_ylabel('Losses')
    plot.tick_params(labelright=True, right=True)
    
    plot.legend(loc="upper right")
    plot.set_title("Average losses over all batches vs Epochs")
    
    figure.tight_layout()
    # plt.show()
    
    # save plot as png
    figure.savefig(path+fig_name+'.png',bbox_inches='tight',dpi=100)
    plt.close(figure)
    
    #return(path+fig_name)
    
    

### Take in tensors (output and target) and construct accuracy map, save as png

def NCHW_to_NHW_np(tensor):
    '''Converts NCHW tensor to numpy NHW array'''
    tensor_numpy = tensor.cpu().detach().numpy()
    tensor_numpy = np.transpose(tensor_numpy, axes=(0,3,1,2)).squeeze()
    return(tensor_numpy)

def acc_map(output, target, clip=1, clip_bad=1):
    '''output and target are NHW numpy arrays, this function computes the accuracy map and
    returns it in format NHW.'''
    diff = np.clip( np.abs( output-target ) ,0 , clip)
    acc  = np.divide(diff,np.abs(target))
    acc  = np.clip(acc,0,clip_bad)
    acc *= (-1/clip_bad)    # normalise to -1 -> 0 with clipping at value=clip_bad for worst results
    acc += 1    # note that acc is in 0-1 range, 1 corresponds to perfect accuracy, 
                # 0 is arbitrary bad acc (if clip_bad=1, corresponds to 100% relative error)
    return(acc)
    
def plot_multi_acc_map(output, target, clip=1, clip_bad=1, fig_name="accuracy_map_fig", path=os.getcwd()):
    '''returns acc_map of shape NHW, while saving the corresponding figure.
    clip: corresponds to the clipping of max values in absolute difference 
    between output and target (1e-8 for example).
    clip_bad: corresponds to the clipping of worst relative error percentage,
    if clip_bad = 1 then worst offset is 100% from target value, if clip_bad=2 then
    worst offset is 200% from target value. This is forst quality of display mainly.'''
    n = output.shape[0]
    out_np  = NCHW_to_NHW_np(output)
    targ_np = NCHW_to_NHW_np(target)
    acc_np  = acc_map(out_np, targ_np, clip=clip, clip_bad=clip_bad)
    acc_min,acc_max = acc_np.min(),acc_np.max()
    
    fig, axes = plt.subplots(nrows=3, ncols=n, constrained_layout=True)
    fig.set_size_inches((12.0, 12.0), forward=False)
    for col in range(n):
        # output
        imOut  = axes[0,col].imshow(out_np[col], cmap='gray', aspect='auto', interpolation='none')
        # target
        imTarg = axes[1,col].imshow(targ_np[col], cmap='gray', aspect='auto', interpolation='none')
        # accuracy map
        imAcc  = axes[2,col].imshow(acc_np[col], vmin=acc_min, vmax=acc_max, aspect='auto',interpolation='none')
        for row in range(3):
            axes[row,col].tick_params(left = False, right = False , labelleft = False ,
                                     labelright = False, labelbottom = False, bottom = False)
    fig.colorbar(imAcc, ax=axes.ravel().tolist(), label="accuracy",
                 orientation = "horizontal", pad = 0.01, aspect=40)
    fig.suptitle("Accuracy map of denoiser output")
    # plt.show()
    
    fig.savefig(path+fig_name+'.png',bbox_inches='tight',dpi=100)
    plt.close(fig)
    
    return(acc_np)
    

    
### Take in tensors (output, target, input) and construct comparison map (scramble): input, target, output-target diff. show extracted noise?
def plot_multi_comp_map(output, target, fig_name="comparison_map_fig", path=os.getcwd()):
    '''returns comparison_map of shape NHW, while saving the corresponding figure'''
    n = output.shape[0]
    out_np  = NCHW_to_NHW_np(output)
    targ_np = NCHW_to_NHW_np(target)
    comp_np = out_np - targ_np
    comp_min,comp_max = comp_np.min(),comp_np.max()
    
    fig, axes = plt.subplots(nrows=3, ncols=n, constrained_layout=True)
    fig.set_size_inches((12.0, 12.0), forward=False)
    for col in range(n):
        # output
        imOut  = axes[0,col].imshow(out_np[col], cmap='gray', aspect='auto', interpolation='none')
        # target
        imTarg = axes[1,col].imshow(targ_np[col], cmap='gray', aspect='auto', interpolation='none')
        # accuracy map
        imComp = axes[2,col].imshow(comp_np[col], vmin=comp_min, vmax=comp_max, aspect='auto', interpolation='none')
        for row in range(3):
            axes[row,col].tick_params(left = False, right = False , labelleft = False ,
                                     labelright = False, labelbottom = False, bottom = False)
    fig.colorbar(imComp, ax=axes.ravel().tolist(), label="difference",
                 orientation = "horizontal", pad = 0.01, aspect=40)
    fig.suptitle("Comparison map of denoiser output and target")
    # plt.show()
    
    fig.savefig(path+fig_name+'.png',bbox_inches='tight',dpi=100)
    plt.close(fig)
    
    return(comp_np)


### Take in specific losses (BCE, KLD, SL) and plot for each epoch, include relative importance in global loss (with coeffs alpha beta gamma)

def plot_different_losses(loss_array, fig_name="different_losses_fig", path=os.getcwd()):
    '''plots different components of loss function in loss_array and saves the corresponding figure.
    Note that loss_array dimensions are: (#_of_components, 2or3, #_of_epochs). 2or3 because 2nd dim entry is 
    the name of the component: [ ['BCE',[0.12, 0.33, ...]] , ['KLD',[0.43, 0.84, ...]] , ...]. 3 because we could add  '''
    
    n_comp  = loss_array.shape[0]
    n_epoch = loss_array[0][1].shape[0]
    # erase   = min(n_epoch,5)-1
    val_too = loss_array.shape[1]==3 # ['name_of_comp', [train_loss], [val_loss]] val loss is optional
    
    fig, axes = plt.subplots(nrows=1, ncols=n_comp)
    fig.set_size_inches((12.0, 12.0), forward=False)
    colors = cm.rainbow(np.linspace(0,1,n_comp))
    for col in range(n_comp):
        axes[col].plot(np.arange(0,n_epoch,step=1), loss_array[col,1], 
                       c=colors[col], label=loss_array[col,0]+' train')
        # axes[col].set_ylim(top=loss_array[col,1][erase],
        #                        bottom = 0.9*min(loss_array[col,1]))
        up_bound_95 = np.quantile(loss_array[col,1],q=0.85)
        low_bound   = 0.85 * loss_array[col,1].min()
        axes[col].set_ylim(bottom=low_bound,
                      top=up_bound_95)
        #if validation loss, plot as well
        if val_too: 
            axes[col].plot(np.arange(0,n_epoch,step=1), loss_array[col,2], '-.', 
                                   c=colors[col], label=loss_array[col,0]+' val')
            up_bound_95 = max(np.quantile(loss_array[col,1],q=0.85), 
                              np.quantile(loss_array[col,2],q=0.85))
            low_bound   = 0.85 * min(loss_array[col,1].min(), 
                                     loss_array[col,2].min())
            axes[col].set_ylim(bottom=low_bound,
                          top=up_bound_95)
            # axes[col].set_ylim(top=loss_array[col,1][erase],
            #                    bottom = 0.9*min(min(loss_array[col,1]),min(loss_array[col,2])))
        axes[col].set_title(loss_array[col,0] + ' loss')
        axes[col].tick_params(right=True,labelright=True)
        axes[col].set_ylabel('Loss')
        axes[col].set_xlabel('Epochs')
        axes[col].legend(loc='upper right')

    fig.suptitle('Loss function components vs Epochs')
    fig.tight_layout()
    # plt.show()
    
    fig.savefig(path+fig_name+'.png',bbox_inches='tight',dpi=100)
    plt.close(fig)
    # return(path+figname)


