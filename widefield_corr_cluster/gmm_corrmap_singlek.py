#!/shared/utils.x86_64/python-2.7/bin/python

"""
# -*- coding: utf-8 -*-
Created on Fri Nov 21 17:53:22 2014

@author: mattv
"""

import time
import os
import numpy as np
import platform
import datetime
import sys
if platform.system() == 'Linux': 
    sys.path.append(r'/data/nc-ophys/BehaviorSoftwareStaging/aibs/') # if windows local machine this should already be in base path
    import matplotlib
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn import mixture
import CorticalMapping.tifffile as tiff
from matplotlib import colors
#if platform.system() == 'Linux': 
    #os.system("taskset -p 0xff %d" % os.getpid()) #reset task affinity after module import
    # reference: http://stackoverflow.com/questions/15639779/what-determines-whether-different-python-processes-are-assigned-to-the-same-or-d
    
def mask_to_index(mask):
    mask = mask.astype(np.intp) #64 bit integer
    mask_idx = np.ndarray.nonzero(mask)
    pullmask = mask.reshape(mask.shape[0]*mask.shape[1]) #flatten image to vector
    pullmask = np.squeeze(np.array([pullmask==1])) #convert to boolean
    pushmask = np.ndarray.nonzero(pullmask) # indexes for within the mask 
    pushmask = pushmask[0]
    return mask_idx, pullmask, pushmask
    
def make_train_test(all_corr): # generate test/train sets choosing random samples (not dimensions!) without replacement
    #rand = np.random.randint(0, len_iter, size=np.ceil(len_iter*.1)) #take random 10% 
    rand = np.random.choice(len_iter, size=np.ceil(len_iter*takerand), replace=None)
    rand_inv = np.ones(len_iter, 'i4')
    rand_inv[rand] = 0
    rand_inv = np.nonzero(rand_inv==1)
    all_corr_test = all_corr[rand,:]
    all_corr_train = all_corr[rand_inv,:] #take random 90% for train
    all_corr_train = all_corr_train.squeeze()
    return all_corr_test, all_corr_train, rand
    
def smart_imshow_cscale(plot_probs):
    plot_probs_lin = np.reshape(plot_probs, [masky*maskx])
    #img_std = np.nanstd(plot_probs_lin)
    img_median = np.nanmedian(plot_probs_lin)
    img_mad = np.nanmedian(np.absolute(plot_probs_lin - np.nanmedian(plot_probs_lin))) #mean absolute deviation
    upper_clim = img_median+(img_mad*3)
    lower_clim = img_median-(img_mad*3)
    return lower_clim, upper_clim
    
def Calinski_Harabasz(means, k): # The Calinski_Harabasz Criterion (variance ratio)
    # http://www.mathworks.com/help/stats/clustering.evaluation.calinskiharabaszevaluation-class.html
    num_obs = pushmask.shape[0]
    overall_mean = np.mean(np.mean(means, axis=1))
    overall_mean_vec = np.tile(overall_mean, k)
    clust_centroid = np.mean(means, axis=1) #all cluster centroids
    to_norm = np.array([clust_centroid, overall_mean_vec]) #conbine centroids with repeating values of the overall mean
    btwn_var = np.sum(np.linalg.norm(to_norm, axis=0)) # sum of euclidean distances between centroids and overall mean
    get_dist = np.zeros((k, pushmask.shape[0]))
    for x in range(k):
        for n in range(num_obs): # for each sample
            get_euclidian = np.array([means[x,n], clust_centroid[x]])
            get_dist[x,n] = np.linalg.norm(get_euclidian)
    within_var = np.sum(np.sum(get_dist, axis=1)) 
    ch = (np.mean(btwn_var)/np.mean(within_var))*(num_obs-k/(k-1)) #for single observation
    return ch
    
#SAVEOUT = True
do_plot = True
do_gmm = True
do_dpgmm = False
# define number of clusters k
k = 6
num_testreps = 5 # test model this many times to build distribution of data probabilities
takerand = .5
gmm_type = 'diag'

filename = '141204JCamF106deci244_cormap_tx10_sx8_2014-12-25.npy'
#filename = '141204JCamF106deci244_cormap_tx10_sx4_full_2015-01-26.npy'
#filename = '20150221JCamF100deci244_cormap_tx2_sx4_2015-02-21.npy'
#filename = '141204JCamF106deci244_cormap_64_subx2562014-12-08.npy'
filename_short = '141204JCamF106deci244_cormap_tx10_sx8'
#filename_short = '141204JCamF106deci244_cormap_tx10_sx4'
special = '_k=' + str(k) + '_' + gmm_type + '_gmm_' # special string to add to filename

if platform.system() == 'Windows':
    #basepath = r'Z:\Matt\141030-M152529\141030JCam130_cormap_2014-11-30.npy'
    #basepath = r'C:\Users\mattv\Desktop\local data\141204'  
    basepath = r'\\aibsdata2\nc-ophys\Bethanny\141204'
    #basepath = r'\\aibsdata2\nc-ophys\Matt\corrdata\150221_M165712'
    #basepath = r'\\aibsdata2\nc-ophys\Matt\corrdata\Doug_150110-M154265'    
    #basepath = r'\\aibsdata2\nc-ophys\Matt\corrdata\Jun_141120-M146303'
elif platform.system() == 'Linux':
    basepath = r'/data/nc-ophys/Bethanny/141204/'
elif platform.system() =='Darwin':
    basepath = r'/Volumes/nc-ophys/Bethanny/141204/'
    
maskname = '141204JCamF106deci244_dffmovie_64_mask.tif' # this should match teh ultimate size of the correlation matrix
#maskname = '128_mask.tif'
maskpath = os.path.join(basepath,maskname)
corr_path = os.path.join(basepath, 'correlations')
filepath = os.path.join(corr_path,filename)
gmm_path = os.path.join(basepath, 'gmm_cluster')
timestamp_str = str(datetime.datetime.now()) # this may be added to save strings later

print filepath
print maskname
print special

color_defs = ["#e50000", "#7b002c", "#6d5acf", "#de0c62", "#feb308",
              "#d8863b", "#e2ca76", "#a00498",  "#01889f", "#7b5804",
              "#c5c9c7", "#7a687f", "#ffff14", "#0504aa", "#9af764",
              "#ff69af", "#536267"]
color_defs = color_defs+color_defs+color_defs+color_defs+color_defs
colors = [colors.colorConverter.to_rgba(i) for i in color_defs]

# get data
mask = tiff.imread(maskpath) # mask must be 8bit thresholded as 0 & 255
mask = mask/255
all_corr = np.load(filepath, 'r+') #load numpy array 
all_corr[np.isnan(all_corr)] = 0.0
#whiten correlation data
all_corr = preprocessing.scale(all_corr)

# get mask indices
masky = mask.shape[0]
maskx = mask.shape[1]
mask_idx, pullmask, pushmask = mask_to_index(mask)

len_iter = all_corr.shape[0] #all_corr shoudl be square correlation matrix with flattened space dimensions taken from the mask
if len_iter != pushmask.shape[0]:
    print 'mask indexing error'

# pre-allocate some things
all_prob_samples = np.zeros([all_corr.shape[0], num_testreps])
all_prob_samples[:] = np.nan
all_respos = np.zeros([all_corr.shape[0], k, num_testreps])
#all_respos[:] = np.nan
plot_probs = np.zeros([masky*maskx])
plot_probs[:] = np.nan
plot_respos = np.zeros([masky*maskx,k])
plot_respos[:] = np.nan

# Initialize model
if do_gmm:
    timer = time.time()
    means = np.zeros([num_testreps, all_corr.shape[0]])
    all_corr_test, all_corr_train, test_idxes = make_train_test(all_corr)
    gmm = mixture.GMM(n_components=k,
        covariance_type=gmm_type, #full takes ~10x longer than diag
        random_state=None,
        thresh=0.01,
        min_covar=0.001,
        n_iter=100,
        n_init=1,
        params='wmc',
        init_params='wmc')
    #gmm.fit(all_corr_train)
    #means = gmm.means_
        
    # train and test num_testreps times
    aic = np.zeros([num_testreps])
    bic = np.zeros([num_testreps])
    rep_means = np.zeros([k, all_corr.shape[0], num_testreps])
    weights = np.zeros([k,num_testreps])
    for i in range(num_testreps):
        all_corr_test, all_corr_train, test_idxes = make_train_test(all_corr)
        gmm.fit(all_corr_train)
        rep_means[:,:,i] = gmm.means_
        means = gmm.means_   
        weights[:,i] = gmm.weights_
        # test model; dim=0 are the log_probs, dim=1 are responsabilities
        all_prob_samples[test_idxes,i], all_respos[test_idxes,:,i] = gmm.score_samples(all_corr_test)
        bic[i] = gmm.bic(all_corr_test)     
        aic[i] = gmm.aic(all_corr_test)
        print 'testing ', i
    aic_mean = np.mean(aic)
    bic_mean = np.mean(bic)
    ch = Calinski_Harabasz(means, k)
    print 'Model converged? = ', gmm.converged_
    print 'BIC = ', bic_mean   
    print 'AIC = ', aic_mean
    print 'Calinski Harabasz criterion = ', ch
    print 'Model testing at best k (s)', time.time() - timer
        
elif do_dpgmm: #dirlicht gmm
    timer = time.time()
    means = np.zeros([num_testreps, all_corr.shape[0]])
    all_corr_test, all_corr_train, test_idxes = make_train_test(all_corr)
    gmm = mixture.DPGMM(n_components=k,
        covariance_type=gmm_type, #full takes ~10x longer than diag
        alpha=.001,
        thresh=0.01,
        min_covar=0.001,
        n_iter=100,
        params='wmc',
        init_params='wmc')

    # train and test num_testreps times
    bic = np.zeros([num_testreps])
    rep_means = np.zeros([k, all_corr.shape[0], num_testreps])
    weights = np.zeros([k,num_testreps])
    for i in range(num_testreps):
        all_corr_test, all_corr_train, test_idxes = make_train_test(all_corr)
        gmm.fit(all_corr_train)
        rep_means[:,:,i] = gmm.means_
        means = gmm.means_
        weights[:,i] = gmm.weights_
        # test model; dim=0 are the log_probs, dim=1 are responsabilities
        all_prob_samples[test_idxes,i], all_respos[test_idxes,:,i] = gmm.score_samples(all_corr_test)
        bic[i] = gmm.bic(all_corr_test)   
        print 'testing ', i
    bic_mean = np.mean(bic)
    print 'Model converged? = ', gmm.converged_
    print 'Num clusters = ', gmm.means_.shape[0]
    print 'BIC = ', bic_mean    
    print 'Model testing at best k (s)', time.time() - timer


### find cluster assignments by grouping components according to correlation coeffs
min_dif = np.zeros([k,num_testreps])
clust_dif = np.zeros([num_testreps,k*2,k*2])
rep_means_swap = rep_means.swapaxes(1,2).swapaxes(0,1)
first_clust = rep_means_swap[0,:,:]
for i in range(num_testreps):
    clust_dif[i,:,:] = np.corrcoef(first_clust, rep_means_swap[i,:,:]) # k by k matrix of corr coeffs
clust_dif_trunc = clust_dif[:,-k:,-k:] #take lower right quadrant of covar matrix
for i in range(num_testreps):
    for x in range(k):
        sort_idx = np.argsort(clust_dif_trunc[i,x,:]) #get cluster index with largest corr coeff.
        min_dif[x,i] = sort_idx[k-2] # Largest corr coef is the second last sorted value, last value is unity correlation
min_dif[:,0] = np.arange(k) #kludge: set to own index. Corr of first dimension is an autocorrelation so it is improper to search for second largest value
min_dif = min_dif.astype(np.intp)

### get max gmm mean at each pixel
kmax_mask = np.zeros([k, means.shape[1]], dtype=('int32'))
kmax_val = np.zeros([means.shape[1]], dtype=('float'))
gmm_kmax = np.empty([k, masky*maskx])* np.nan
for i in range(means.shape[1]): #for each pixel
    kmax_val[i] = means[np.argmax(means[:,i]), i]
    kmax_mask[np.argmax(means[:,i]), i] = 1
kmax_val_img = np.empty([masky*maskx])* np.nan
kmax_val_img[pushmask] = kmax_val
   
# re-format outputs back into a square image
# pre-allocate means image as NaNs so mask background has no value
gmm_means = np.zeros([k, masky*maskx])
gmm_means[:] = np.nan
for n in range(k):
    gmm_means[n, pushmask] = means[n, :]

# rearrange the cluster order of the feature data according to clusters with 2nd highest correlation coeffs
aranged_respos = np.zeros([all_corr.shape[0], k, num_testreps])
for r in range(num_testreps):
    for n in range(k):
        aranged_respos[:,n,r] = all_respos[:, min_dif[n,r], r]

prob_samples = np.nanmean(all_prob_samples, axis=1)
plot_probs[pushmask] = prob_samples   
plot_probs = plot_probs.reshape([masky,maskx])
#prob_features = np.nanmean(all_respos, axis=2)
respos = np.nanmean(aranged_respos, axis=2) #average responsibilities over tests
plot_respos[pushmask,:] = respos
plot_respos = plot_respos.reshape([masky,maskx,k])
   
    
#
#if SAVEOUT: 
#    savepath1 = os.path.join(savepath,filename_short+'gmm_model_'+special)
#    gmmoutput = open(savepath1, 'w')
#    pickle.dump(gmm, gmmoutput)
#

#############################################################################
### PLOTTING    

if do_plot:
    minprob, maxprob = smart_imshow_cscale(plot_probs)
    minmean, maxmean = smart_imshow_cscale(np.reshape(kmax_val_img, [masky, maskx]))
    xmin, xmax, ymin, ymax = [0, maskx, 0, masky]
    extent = xmin, xmax, ymin, ymax
    
    ### All means
    #minmean = np.nanmin(np.nanmin(gmm_means, axis=1), axis=0)
    #maxmean = np.nanmax(np.nanmax(gmm_means, axis=1), axis=0)
    fig1 = plt.figure(figsize = (12,4))
    fig1.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=.05, wspace=.05)
    for i in range(k):
        ax1 = fig1.add_subplot(2, np.ceil(k/2.), i+1, xticks=[], yticks=[])
        ax1.imshow(gmm_means[i,:].reshape(masky,maskx), cmap='gnuplot2', 
                   interpolation='nearest', vmin = minmean, vmax = maxmean)           
    figpath1 = os.path.join(gmm_path,filename_short+'gmm_means_'+special+timestamp_str[:10])
    plt.savefig(figpath1,  dpi=300, transparent=True)
    
    ### Max Means
    from matplotlib import colors
    fig2 = plt.figure(figsize = (4,4), frameon=False);
    ax2 = fig2.add_subplot(1,1,1, xticks=[], yticks=[]);
    for n in range(k):
        gmm_kmax[n, pushmask] = kmax_mask[n, :]    
        cmap = colors.ListedColormap([[0,0,0,0], color_defs[n]])
        plt.imshow(np.reshape(gmm_kmax[n,:], [masky, maskx]), cmap=cmap, alpha=1, extent=extent)
    plt.imshow(np.reshape(kmax_val_img, [masky, maskx]), cmap='gray', alpha=0.5, extent=extent, vmin = minmean, vmax = maxmean)
    figpath2 = os.path.join(gmm_path,filename_short+'gmm_maxmeans_'+special+timestamp_str[:10])
    plt.savefig(figpath2,  dpi=300, transparent=True)
        
    ### tested likelyhoods per cluster   
    fig6 = plt.figure(figsize = (13,4));
    fig6.subplots_adjust(left=0, right=1, bottom=0, top=0.9, hspace=.05, wspace=.05);
    fig6.suptitle('Data likelyhood in each class responsibility', fontsize=14);
    
    for i in range(k):
        ax6 = fig6.add_subplot(2, np.ceil(k/2.), i+1, xticks=[], yticks=[]);
        a6 = ax6.imshow(mask, interpolation='nearest', alpha=0.1, cmap='gray_r')
        a6 = ax6.imshow(plot_respos[:,:,i], 
                        interpolation='nearest', 
                        vmin=minprob, vmax=maxprob, cmap='RdBu_r');
    
    coloraxis = fig6.add_axes([1, 0.1, 0.03, 0.8]);
    plt.colorbar(a6, cax=coloraxis)
    figpath6 = os.path.join(gmm_path,filename_short+'data_likelyhood_per_responsibility'+special+timestamp_str[:10])
    plt.savefig(figpath6, dpi=300, transparent=True)
    
    ### aggregated likelyhoods from all clusters
    fig7 = plt.figure(figsize = (4,4));
    fig7.subplots_adjust(left=0, right=1, bottom=0, top=0.9, hspace=.05, wspace=.05);
    fig7.suptitle('Likelyhood of each datapoint under the model', fontsize=14);
    ax7 = fig7.add_subplot(1,1,1, xticks=[], yticks=[]);
    a7 = ax7.imshow(mask, interpolation='nearest', alpha=0.1, cmap='gray_r')
    a7 = ax7.imshow(plot_probs, 
                    interpolation='nearest', 
                    vmin=minprob, vmax=maxprob, cmap='RdBu_r');
    coloraxis = fig7.add_axes([1, 0.1, 0.03, 0.8]);
    plt.colorbar(a7, cax=coloraxis)
    figpath7 = os.path.join(gmm_path,filename_short+'data_likelyhood_'+special+timestamp_str[:10])
    plt.savefig(figpath7,  dpi=300, transparent=True)