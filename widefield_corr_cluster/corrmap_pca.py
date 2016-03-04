#!/shared/utils.x86_64/python-2.7/bin/python

"""
Created on Mon Feb 23 12:34:22 2015

"...this code is like being in a house built by a child using nothing but
a hatchet and a picture of a house"
    - xkcd, 1513
    
This code does a varied set of analyses starting with the full seed-pixel correaltion 
matrix of widefield-imaged neural activity

Classes of analyses (not yet implemented as pythonic clases):
1. Apply mulitple thresholds to correlation matrix, find centroids of thresholded correlation masks, 
    cluster centroids, reconstruct correlation matrix using only seed-pixels that produce
    clustered centroids.
2. dimensionality reduction of corr matrix using PCA
3. ICA component seperation
4. find average correlation over space for brain or brain region
5. Gaussiam mixture model (GMM) or Dirichlet model building
6. GMM model analysis using Bayesian infromation criteria

@author: mattv
"""

import numpy as np
import platform
import os
import sys
import datetime
if platform.system() == 'Linux':
    import matplotlib
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
if platform.system() == 'Linux': 
    sys.path.append(r'/data/nc-ophys/Matt/aibs/') # if windows local machine this should already be in base path 
import aibs.CorticalMapping.tifffile as tiff
from sklearn.decomposition import RandomizedPCA
from sklearn.decomposition import FastICA
from sklearn import mixture
import scipy.stats as stat
from scipy import linalg, dot
from skimage import feature
from sklearn.cluster import DBSCAN
import operator
import copy
import h5py
import pdb
import pickle


if platform.system() == 'Windows':
    basepath = r'E:\local data\150225_M166910'
    #basepath = r'\\aibsdata2\nc-ophys\Bethanny\141204'
    #basepath = r'C:\Users\mattv\Desktop\local data\150225_M166910'
    filename = '20150225JCamF102deci244_cormap_tx100_sx8_lag_0_2015-04-27.npy'
    maskname = '102mask_64.tif'
    masknameleft = '102mask_64_left.tif'
    masknameright = '102mask_64_right.tif'
    filename_short = '0150225JCamF102deci244_cormap_tx10_sx2'
    
#if platform.system() == 'Windows':
#    basepath = r'\\aibsdata2\nc-ophys\Matt\2P_neuropil_subtraction'
#    filename = 'cormap_tx1'
#    maskname = 'mask152.tif'
#    masknameleft = '102mask_128_left.tif'
#    masknameright = '102mask_128_right.tif'
#    filename_short = 'cormap_tx1'
    
elif platform.system() == 'Linux':
    basepath = r'/data/nc-ophys/Matt/corrdata/M177929/20150701'
    filename = '20150701JCamF101deci1044_cormap_tx10_sx1_lag_0_2015-07-02'
    maskname = '102mask_128.tif'
    masknameleft = '102mask_128_left.tif'
    masknameright = '102mask_128_right.tif'
    filename_short = '20150701JCamF101deci1044_cormap_tx10_sx1'
   
pca_path = os.path.join(basepath, 'pca')
corr_path = os.path.join(basepath, 'correlations')
gmm_path = os.path.join(basepath, 'gmm_cluster')
maskpath = os.path.join(basepath,maskname)
filepath = os.path.join(corr_path,filename)
maskpathleft = os.path.join(basepath,masknameleft)
maskpathright = os.path.join(basepath,masknameright)

#GLOBALS
pc_start = 5 # 0 is first pc
num_ica = 10
takerand = .2
num_testreps = 10 # test model this many times to build distribution of data probabilities
psf_width = 30
rsamp_factor = 1
gmm_type = 'spherical'
SAVE_PLOT = False
do_psf = False
do_thresh_corr = True #before GMM building, make sum-projection of thresholded correlation maps
scan_pcas = False
single_pca = False
do_pcaica = False
do_dpgmm = False # if false do normal gmm, not Dirichlet
symmetrical_test_train = False # if false, this fraction will be tested, the rest used for training
hdf5_format = False

timestamp_str = str(datetime.datetime.now())
   

'''  '''
'''  '''   
     
   
def tableau20_colors(color_to_get):
    colors = ((0,0,0),
                (31,119,180),                
                (255,127,14),
                (44,160,44),
                (214,39,40),
                (148,103,189),
                (255,152,150),
                (174,199,232),
                (255,187,120),
                (197,176,213),
                (152,223,138),                
                (247,182,210),
                (227,119,194),
                (196,156,148),
                (140,86,75),
                (127,127,127),
                (219,219,141),
                (199,199,199),
                (188,189,34),
                (158,218,229),
                (23,190,207),
                (31,119,180),                
                (255,127,14),
                (44,160,44),
                (214,39,40),
                (148,103,189),
                (255,152,150),
                (174,199,232),
                (255,187,120),
                (197,176,213),
                (152,223,138),                
                (247,182,210),
                (227,119,194),
                (196,156,148),
                (140,86,75),
                (127,127,127),
                (219,219,141),
                (199,199,199),
                (188,189,34),
                (158,218,229),
                (23,190,207))
    return colors[color_to_get]

def make_train_test(dat): # generate test/train sets choosing random samples (not dimensions!) without replacement
    #rand = np.random.randint(0, len_iter, size=np.ceil(len_iter*.1)) #take random 10% 
    rand = np.random.choice(dat.shape[0], size=np.ceil(dat.shape[0]*takerand), replace=False)
    rand_inv = np.ones(dat.shape[0], 'i4')
    rand_inv[rand] = 0
    rand_inv = np.nonzero(rand_inv==1)
    all_corr_test = dat[rand,:]
    all_corr_train = dat[rand_inv,:] #take complement of random for train
    all_corr_train = all_corr_train.squeeze()
    return all_corr_test, all_corr_train, rand, rand_inv
    
def make_train_test_sym(dat): # generate test/train sets choosing random samples (not dimensions!) without replacement. Choose equal sized amounts of data for tarin and test not exceeding 50% (obviously)
    if takerand > 0.5:
        print 'ERROR, reduce "takerand" to below 0.5'    
        sys.exit()
    rand_test = np.random.choice(dat.shape[0], size=np.ceil(dat.shape[0]*takerand), replace=False)    
    rand_train = np.random.choice(dat.shape[0], size=np.ceil(dat.shape[0]*takerand), replace=False)
    all_corr_test = dat[rand_test,:]
    all_corr_train = dat[rand_train,:]
    return all_corr_test, all_corr_train, rand_test, rand_train
    
def mask_to_index(mask):
    mask = mask.astype(np.intp) #64 bit integer
    mask_idx = np.nonzero(mask)
    pullmask = mask.reshape(mask.shape[0]*mask.shape[1]) #flatten image to vector
    pullmask = np.squeeze(np.array([pullmask==1])) #convert to boolean
    pushmask = np.nonzero(pullmask) # indexes for within the mask 
    pushmask = pushmask[0]
    return mask_idx, pullmask, pushmask
    
def smart_imshow_cscale(dat):
    if dat.ndim == 2:
        dat = np.reshape(dat, [dat.shape[0]*dat.shape[1]])
    elif dat.ndim == 3:
        dat = np.reshape(dat, [dat.shape[0]*dat.shape[1]*dat.shape[2]])
    img_median = np.nanmedian(dat)
    img_mad = np.nanmedian(np.absolute(dat - np.nanmedian(dat))) #mean absolute deviation
    upper_clim = img_median+(img_mad*4)
    lower_clim = img_median-(img_mad*4)
    return lower_clim, upper_clim
    
def plot_all_clusters(dat):    
    fig1 = plt.figure(figsize = (12,4))
    fig1.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=.05, wspace=.05)
    for i in range(dat.shape[0]):
        minmap, maxmap = smart_imshow_cscale(dat[i,:])
        ax1 = fig1.add_subplot(4, np.ceil(dat.shape[0]/4.), i+1, xticks=[], yticks=[])
        ax1.axis('off')
        ax1.imshow(dat[i,:], cmap='gnuplot2', 
               interpolation='nearest', vmin = minmap, vmax = maxmap) 
    if SAVE_PLOT:
        special = '_k=' + str(k) + '_' + gmm_type + '_gmm_pca=' + str(pc_end) + '_randsamp=' + str(rsamp_factor) + '_'
        figpath = os.path.join(gmm_path,filename_short+'_ICA_'+special+timestamp_str[:10])
        plt.savefig(figpath,  dpi=300, transparent=True)
               
def plot_all_masks(dat):    
    fig1 = plt.figure(figsize = (12,4))
    fig1.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=.05, wspace=.05)
    for i in range(dat.shape[0]):
        plot_dat = copy.copy(mask)
        plot_dat[dat[i,:,:]] = 4
        ax1 = fig1.add_subplot(4, np.ceil(dat.shape[0]/4.), i+1, xticks=[], yticks=[])
        ax1.imshow(plot_dat, cmap='Greys', interpolation='nearest')               

def Calinski_Harabasz(means, k): # The Calinski_Harabasz Criterion (variance ratio)
    # http://www.mathworks.com/help/stats/clustering.evaluation.calinskiharabaszevaluation-class.html
    num_obs = means.shape[1]
    overall_mean = np.mean(np.mean(means, axis=1))
    overall_mean_vec = np.tile(overall_mean, k)
    clust_centroid = np.mean(means, axis=1) #all cluster centroids
    to_norm = np.array([clust_centroid, overall_mean_vec]) #conbine centroids with repeating values of the overall mean
    btwn_var = np.sum(np.linalg.norm(to_norm, axis=0)) # sum of euclidean distances between centroids and overall mean
    get_dist = np.zeros((k, num_obs))
    for x in range(k):
        for n in range(num_obs): # for each sample
            get_euclidian = np.array([means[x,n], clust_centroid[x]])
            get_dist[x,n] = np.linalg.norm(get_euclidian)
    within_var = np.sum(np.sum(get_dist, axis=1)) 
    ch = (np.mean(btwn_var)/np.mean(within_var))*(num_obs-k/(k-1)) 
    return ch
    
def correlation_psf(all_corr_mov, mask_idx, psf_width):
    #kernal = np.ones([10,10])
    local_corr = np.zeros([all_corr_mov.shape[0], psf_width*2, psf_width*2])
    local_corr[:] = np.nan
    for n in range(all_corr_mov.shape[0]):
        yidx = np.array(mask_idx[0])
        xidx = np.array(mask_idx[1])
        ymax = all_corr_mov.shape[1]
        xmax = all_corr_mov.shape[2]
        if np.min(yidx[n])-psf_width < 0 or np.min(xidx[n])-psf_width < 0 or np.max(yidx[n])+psf_width > ymax or np.max(xidx[n])+psf_width > xmax:
            local_corr[n,:,:] = np.nan
        else:
            local_corr[n,:,:] = all_corr_mov[n, yidx[n]-psf_width:yidx[n]+psf_width, xidx[n]-psf_width:xidx[n]+psf_width]
    return local_corr
    
def train_test_gmm(gmm, gmm_dat, k):
    all_prob_samples = np.zeros([gmm_dat.shape[0], num_testreps])
    all_prob_samples[:] = np.nan
    all_respos = np.zeros([gmm_dat.shape[0], k, num_testreps])
    all_respos[:] = np.nan
    resp_matched = np.zeros([gmm_dat.shape[0], k, num_testreps])
    resp_matched[:] = np.nan
    aic = np.zeros([num_testreps])
    bic = np.zeros([num_testreps])
    #idx = np.zeros([k,num_testreps])
    for i in range(num_testreps):
        if symmetrical_test_train:
            all_corr_test, all_corr_train, test_idxes, train_idxes = make_train_test_sym(gmm_dat)
        else:
            all_corr_test, all_corr_train, test_idxes, train_idxes = make_train_test(gmm_dat)
        gmm.fit(all_corr_train)
        print 'Testing cycle ' + str(i+1)+'/'+str(num_testreps) + ' model converged = ', gmm.converged_       
        all_prob_samples[test_idxes,i], all_respos[test_idxes,:,i] = gmm.score_samples(all_corr_test) 
        bic[i] = gmm.bic(all_corr_test)
        aic[i] = gmm.aic(all_corr_test)
#    if scan_pcas:
#        for i in range(num_testreps): # run the for loop again. This fixes odd bug where idx is overwritten with last iteration. I have no idea why this works        
#            resp_matched[:,:,i], idx[:,i] = match_respo_pred(k, gmm_dat, all_respos[:,:,i])
    aic_mean = np.mean(aic)
    bic_mean = np.mean(bic)
    #ch = Calinski_Harabasz(means, k)
    num_datapoints = all_corr_test.shape[0]
    print 'Data length with ' + str(takerand) + ' train/test split = ', str(num_datapoints) + ' samples'
    return aic_mean, bic_mean, all_prob_samples, all_respos, num_datapoints

def pca(all_corr, pc_start, pc_end):
    pca_components = pc_end-pc_start
    pca = RandomizedPCA(n_components=pca_components, whiten=False)    
    print 'reducing dimensions to ' + str(pca_components) + ' PCA components'
    pc_idx = range(pc_start, pc_end)      
    pca_xform = pca.fit_transform(all_corr)  
    all_corr_pca = pca_xform[:,pc_idx] #do not whiten PCA-space data       
    eig = pca.components_    
    variances = pca.explained_variance_ratio_
    eigenmaps = np.zeros([pca_components, masky*maskx])
    eigenmaps[:] = np.nan
    eigenmaps[:, pushmask] = eig
    eigenmaps_img = eigenmaps.reshape(pca_components, masky, maskx)
    return eigenmaps_img, all_corr_pca, variances

def build_gmm(k):
    print 'building ' + gmm_type + ' GMM with k=' + str(k) + ' components'
    gmm = mixture.GMM(n_components=k,
        covariance_type=gmm_type, 
        random_state=None,
        thresh=0.01,
        min_covar=0.001,
        n_iter=2000,
        n_init=1,
        params='wmc',
        init_params='wmc')
    return gmm
    
def build_dpgmm(k):
    print 'building ' + gmm_type + ' Dirichlet GMM with k<=' + str(k) + ' components'
    gmm = mixture.DPGMM(n_components=k,
        covariance_type=gmm_type, 
        alpha=1,
        thresh=0.001,
        min_covar=0.001,
        n_iter=500,
        params='wmc',
        init_params='wmc')
    return gmm

def predict_class(all_corr_pca):
    all_predict = gmm.predict(all_corr_pca)
    img_predict = np.zeros([masky*maskx])
    img_predict[:] = np.nan
    img_predict[pushmask] = all_predict
    img_predict = np.reshape(img_predict, [masky, maskx])
    return all_predict, img_predict

def plot_predict(all_corr_pca, k, pc_end, rsamp_factor):
    all_predict, img_predict = predict_class(all_corr_pca)
    fig1 = plt.figure(figsize = (4,4));
    fig1.subplots_adjust(left=0, right=1, bottom=0.1, top=0.9, hspace=.05, wspace=.05);
    ax1 = fig1.add_subplot(1,1,1, xticks=[], yticks=[]);
    fig1.suptitle('Class assignment of every seed-pixel', fontsize=10);
    ax1 = fig1.add_subplot(1,1,1, xticks=[], yticks=[]);
    ax1.text(0.5,0.15,'k=' + str(k) + ', ' + str(pc_end) + ' input PCs', fontsize=9)
    ax1.axis('off')
    ax1.imshow(img_predict, interpolation='nearest')
    if SAVE_PLOT:
        special = '_k=' + str(k) + '_' + gmm_type + '_gmm_pca=' + str(pc_end) + '_randsamp=' + str(rsamp_factor) + '_'
        figpath = os.path.join(gmm_path,filename_short+'_allpredictions_'+special+timestamp_str[:10])
        plt.savefig(figpath,  dpi=300, transparent=True)
        plt.close()

def plot_logprob(all_prob_samples, k, pc_end, rsamp_factor):
    plot_probs = np.zeros([masky*maskx])
    plot_probs[:] = np.nan    
    prob_samples = np.nanmean(all_prob_samples, axis=1)
    plot_probs[pushmask] = prob_samples   
    plot_probs = plot_probs.reshape([masky,maskx])    
    minprob, maxprob = smart_imshow_cscale(plot_probs)
    fig7 = plt.figure(figsize = (4,4));
    fig7.subplots_adjust(left=0, right=1, bottom=0.1, top=0.9, hspace=.05, wspace=.05);
    fig7.suptitle('Likelyhood of each datapoint under the model', fontsize=10);
    ax7 = fig7.add_subplot(1,1,1, xticks=[], yticks=[]);
    ax7.text(0.5,0.15,'k=' + str(k) + ', ' + str(pc_end) + ' input PCs', fontsize=9)
    ax7.axis('off')
    a7 = ax7.imshow(mask, interpolation='nearest', alpha=0.1, cmap='gray_r')
    a7 = ax7.imshow(plot_probs, 
                    interpolation='nearest', 
                    vmin=minprob, vmax=maxprob, cmap='gnuplot2');
    plt.colorbar(a7)
    if SAVE_PLOT:
        special = '_k=' + str(k) + '_' + gmm_type + '_gmm_pca=' + str(pc_end) + '_randsamp=' + str(rsamp_factor) + '_'
        figpath = os.path.join(gmm_path,filename_short+'_data_likelyhood_'+special+timestamp_str[:10])
        plt.savefig(figpath,  dpi=300, transparent=True)
        plt.close()
    
def rand_samp(all_corr, rsamp_factor):
    r_subsamp = np.ceil(all_corr.shape[0]/rsamp_factor)
    r_subsamp = r_subsamp.astype('i32')
    ridx = np.random.randint(0, all_corr.shape[0], r_subsamp)
    all_corr_img = all_corr[ridx, :]
    return all_corr_img, ridx
    
def predict_masks(all_predict, k):
    kmask = np.zeros([k, all_predict.shape[0]], dtype=bool)
    kmask_img = np.zeros([k, masky*maskx], dtype=bool)
    for n in range(k):
        kmask[n,:] = all_predict==n
    kmask_img[:,pushmask] = kmask
    kmask_img = np.reshape(kmask_img, [k, masky, maskx])
    return kmask, kmask_img
    
def all_corr_to_image(all_corr):
    n_samp = all_corr.shape[0]
    dat = np.zeros([n_samp, masky*maskx])
    dat[:] = np.nan
    dat[:,pushmask] = all_corr
    dat = dat.reshape([n_samp, masky, maskx])
    return dat

def mask_to_img(m):
    img = np.zeros([masky*maskx])
    img[:] = np.nan
    img[pushmask] = m
    img = img.reshape([masky,maskx])
    return img
    
def rank_probs(all_prob_samples, kmask, k):
    probs = np.mean(all_prob_samples, 1)
    prob_mean = [np.nanmean(probs[kmask[n,:]]) for n in range(kmask.shape[0])]
    ranked_probs = stat.rankdata(prob_mean)
    ranked_probs = ranked_probs.astype('int32')
    ranked_probs = np.abs(ranked_probs-k) #invert so 0th rank is highest probability
    return ranked_probs
 

def get_DPP(pc_start, pc_end, k, num_datapoints):
    pca_components = pc_end-pc_start
    if gmm_type == 'full':
        params = (pca_components+pca_components**2)*k+(k-1)
        print 'number of parameters = ' + str(params)
        print 'Data per Parameter (DPP) = ' + str(num_datapoints/params)
    elif gmm_type == 'diag':
        params = (pca_components+pca_components)*k+(k-1)
        print 'number of parameters = ' + str(params)
        print 'Data per Parameter (DPP) = ' + str(num_datapoints/params)
    elif gmm_type == 'spherical':
        params = (pca_components+1)*k+(k-1)
        print 'number of parameters = ' + str(params)
        print 'Data per Parameter (DPP) = ' + str(num_datapoints/params)
    
def max_cos_dist(testvec, refvec): # find index of array of reference vectors that has maximum cosine distance to the test vector
    test_copy = np.copy(testvec) #make copy so as not to zap nans in original   
    test_copy[np.isnan(test_copy)] = 0. #zap nans
    dist = [dot(test_copy[:,i],refvec.T)/linalg.norm(test_copy[:,i])/linalg.norm(refvec) for i in range(test_copy.shape[1])]
    max_dist = dist.index(np.max(dist)) 
    return max_dist
    
def match_respo_pred(k, all_corr_pca, resp): #called for each testrep, rearrange the responsibilities to match the predictions   
    all_predict, img_predict = predict_class(all_corr_pca) #get class labels
    predmask, mask_img = predict_masks(all_predict, k)
    predmask = predmask.astype('int32') #turn boolean class assignments into integers    
    indices = [max_cos_dist(resp, predmask[n,:]) for n in range(k)] # for each k get index of the responsibility that best matches the predicted label   
    indices = np.asarray(indices, 'int32') 
    #resp[resp==0] = np.nan   
    rearrange_resp = np.zeros([resp.shape[0], k])
    rearrange_resp[:] = np.nan
    rearrange_resp = resp[:,indices]
    return rearrange_resp, indices
    
def mean_respos(respo): #take the mean responsibility outside of the mask of class assignment
    all_predict, img_predict = predict_class(all_corr_pca) #get class labels
    predmask, mask_img = predict_masks(all_predict, k)
    mask_img = np.rollaxis(mask_img, 0,3)
    #mask_img = np.invert(mask_img)
    mean_residual = respo[mask_img] #take responsibilities outside class mask
    mean_residual = respo #ignore class mask
    mean_residual = np.nanmean(np.nanmean(mean_residual))
    return mean_residual        
        
def plot_mean_respos(respos):
    respos = 1 - (np.squeeze(np.nanmax(respos, axis=1)))
    respos = np.nanmean(respos, axis=1)
    resp_img = np.zeros([masky*maskx])
    resp_img[:] = np.nan
    resp_img[pushmask] = respos
    resp_img = resp_img.reshape([masky,maskx])    
    fig2 = plt.figure(figsize = (4,4))
    fig2.subplots_adjust(left=0, right=1, bottom=0.1, top=0.9, hspace=.05, wspace=.05);
    ax = fig2.add_subplot(1,1,1, xticks=[], yticks=[]);
    fig2.suptitle('Residual responsibilities for every seed-pixel', fontsize=10);
    ax = fig2.add_subplot(1,1,1, xticks=[], yticks=[]);
    ax.text(0.5,0.15,'k=' + str(k) + ', ' + str(pc_end) + ' input PCs', fontsize=9)
    ax = fig2.add_subplot(1, 1, 1, xticks=[], yticks=[])
    ax.axis('off')
    a = ax.imshow(resp_img, cmap='gray_r', interpolation='nearest')
    plt.colorbar(a)
    if SAVE_PLOT:
        special = '_k=' + str(k) + '_' + gmm_type + '_gmm_pca=' + str(pc_end) + '_randsamp=' + str(rsamp_factor) + '_'
        figpath = os.path.join(gmm_path,filename_short+'mean_responsibilities_'+special+timestamp_str[:10])
        plt.savefig(figpath,  dpi=300, transparent=True)
    return resp_img
    
def apply_thresh_corr(dat, corr_thresh): #build masks from correlation matrix, project and normalize masks   
    dat_thresh = np.zeros(dat.shape)
    dat_thresh[dat > corr_thresh] = 1
    dat_thresh[dat <= corr_thresh] = 0
    dat_sum = np.squeeze(np.nansum(dat_thresh, axis=0))
    dat_sum = dat_sum/dat.shape[1] #normalize values so if every seed pixel produces mask at given pixel, value = 1
    proj_threshcorr = mask_to_img(dat_sum)
    return proj_threshcorr
    
def thresh_corr_centroids(dat, maskleft, maskright, corr_thresh):
    
    dat_img = all_corr_to_image(dat) #convert back to image space
    
    fig1 = plt.figure(figsize = (12,4))    
    fig1.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=.05, wspace=.05)
#    fig2 = plt.figure(figsize = (12,4))    
#    fig2.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=.05, wspace=.05)
    #pdb.set_trace()
    l_mask_idx, l_pullmask, l_pushmask = mask_to_index(maskleft)
    r_mask_idx, r_pullmask, r_pushmask = mask_to_index(maskright)
    submask_l = [n for n in range(pushmask.shape[0]) if pushmask[n] in l_pushmask] # indices of pushmask that are also in l_pushmask
    submask_r = [n for n in range(pushmask.shape[0]) if pushmask[n] in r_pushmask]
    lcy = np.zeros([l_pushmask.shape[0]])
    lcx = np.zeros([l_pushmask.shape[0]])
    rcy = np.zeros([r_pushmask.shape[0]])
    rcx = np.zeros([r_pushmask.shape[0]])
    all_centroids_l = np.zeros([len(corr_thresh), l_pushmask.shape[0], 2])
    all_centroids_r = np.zeros([len(corr_thresh), r_pushmask.shape[0], 2])
    
    for t in range(len(corr_thresh)): # for each correlation threshold
        all_centroid_image = np.zeros([masky,maskx])
        for n in range(l_pushmask.shape[0]):            
            lcy[n], lcx[n], idx = get_centroids_from_seeds(dat_img[submask_l[n],:,:],maskleft,corr_thresh[t]) # cy, cx are exact coordinates of centroid, idx is nearest index (rounded towards bottom right pixel)
            all_centroid_image[idx] = all_centroid_image[idx] + 1 
                                                 
        for n in range(r_pushmask.shape[0]):            
            rcy[n], rcx[n], idx = get_centroids_from_seeds(dat_img[submask_r[n],:,:],maskright,corr_thresh[t]) 
            all_centroid_image[idx] = all_centroid_image[idx] + 1
                     
        all_centroids_l[t,:,:] = np.column_stack((lcx, lcy))
        all_centroids_r[t,:,:] = np.column_stack((rcx, rcy))

        all_centroid_image = all_centroid_image*mask # zero regions outside mask
        
        #all_centroids_plot = np.vstack([l_centroid_cores, r_centroid_cores])
        #all_centroids_plot = np.vstack([all_centroids_plot, np.array([[0,0], [0,255], [255,255], [255,0]])])
        
        axcent1 = fig1.add_subplot(3, 5, t+1, xticks=[], yticks=[])
        axcent1.imshow(all_centroid_image, vmax=30, cmap='gray_r', interpolation='nearest') 
        #axcent2 = fig2.add_subplot(3, 5, t+1, xticks=[], yticks=[])
        #axcent2.scatter(all_centroids_plot[:,0], all_centroids_plot[:,1], color='black', marker='.', s=1, edgecolor='none') 
        #plt.gca().invert_xaxis() #scatter plot puts origin in lower left, this inverts y to start in upper left
        del all_centroid_image

    return all_centroids_l, all_centroids_r, submask_l, submask_r
   
def centroids_to_seedpixels(r_centroids, l_centroids, submask_r, submask_l):
    l_seeds = [submask_l[cent] for i,cent in enumerate(l_centroids)]
    r_seeds = [submask_r[cent] for i,cent in enumerate(r_centroids)]
    seedpixels = l_seeds + r_seeds
    return seedpixels
     
def cluster_using_DBSCAN(all_centroids_l, all_centroids_r, maskleft, maskright, min_thresh):
    #fig3 = plt.figure(figsize = (12,4))    
    #fig3.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=.05, wspace=.05)
    #fig4 = plt.figure(figsize = (12,4))    
    #fig4.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=.05, wspace=.05)

    l_mask_idx, l_pullmask, l_pushmask = mask_to_index(maskleft)
    r_mask_idx, r_pullmask, r_pushmask = mask_to_index(maskright)
    submask_l = [n for n in range(pushmask.shape[0]) if pushmask[n] in l_pushmask] # indices of pushmask that are also in l_pushmask
    submask_r = [n for n in range(pushmask.shape[0]) if pushmask[n] in r_pushmask]
    #proj_corr = np.zeros([masky, maskx])
    #pdb.set_trace()
    l_centroid_cores, l_db_cores, l_centroid_labels = dbscan_centroids(all_centroids_l)   
    r_centroid_cores, r_db_cores, r_centroid_labels = dbscan_centroids(all_centroids_r)

    #all_corr_centroids = np.zeros(all_corr.shape)
    #all_corr_centroids[used_seed_pixels_cores,:] = all_corr[used_seed_pixels_cores,:]       
    
    #proj_threshcorr = apply_thresh_corr(all_corr_centroids, min_thresh)
    #proj_corr = apply_thresh_corr(all_corr, min_thresh)
    
    #ax3 = fig3.add_subplot(1, 1, 1, xticks=[], yticks=[])
    #ax3.imshow(proj_corr, cmap='gray_r', interpolation='nearest')  
                    
    #ax4 = fig4.add_subplot(1, 1, 1, xticks=[], yticks=[])   
    #ax4.imshow(proj_threshcorr, cmap='gray_r',  interpolation='nearest')      
    
    
    #l_centroid_labels2 = l_centroid_labels[l_centroid_labels>=0]
    #r_centroid_labels2 = r_centroid_labels[r_centroid_labels>=0]
    used_seed_pixels_cores = centroids_to_seedpixels(r_db_cores, l_db_cores, submask_r, submask_l)
    
    l_centroid_labels = l_centroid_labels.tolist()
    r_centroid_labels = r_centroid_labels.tolist()

    labels_l = [s for i,s in enumerate(set(l_centroid_labels)) if s != -1] # get label categories
    labels_r = [s for i,s in enumerate(set(r_centroid_labels)) if s != -1]  
        
    
    '''
    group seedpixels, core-pixels and centroids according to label
    '''
     
    for n, cat in enumerate(labels_l): 
        #lab_idx = [i for i,ii in enumerate(l_centroid_labels) if ii == cat] # index of all non-noise seed-pixels
        core_idx = [ii for i,ii in enumerate(l_db_cores) if l_centroid_labels[ii] == cat] # index of all core seed-pixels
        label_centroids = all_centroids_l[core_idx]
        label_map = np.zeros([masky*maskx])
        label_map[:] = np.nan
        label_map[l_pushmask[core_idx]] = 1
        #pdb.set_trace()
        if n == 0:
            uber_centroid_l = [(np.ceil(np.nanmean(label_centroids[:,0])), np.ceil(np.nanmean(label_centroids[:,1])))]
            all_label_map = np.copy(label_map)
        else: 
            uber_centroid_l += [(np.ceil(np.nanmean(label_centroids[:,0])), np.ceil(np.nanmean(label_centroids[:,1])))]
            all_label_map = np.vstack((all_label_map, label_map))
            
    for n, cat in enumerate(labels_r): 
        #lab_idx = [i for i,ii in enumerate(r_centroid_labels) if ii == cat] # index of all non-noise seed-pixels
        core_idx = [ii for i,ii in enumerate(r_db_cores) if r_centroid_labels[ii] == cat] # index of all core seed-pixels
        label_centroids = all_centroids_r[core_idx]
        label_map = np.zeros([masky*maskx])
        label_map[:] = np.nan
        label_map[r_pushmask[core_idx]] = 1
        all_label_map = np.vstack((all_label_map, label_map))
        if n == 0:
            uber_centroid_r = [(np.ceil(np.nanmean(label_centroids[:,0])), np.ceil(np.nanmean(label_centroids[:,1])))]
        else: 
            uber_centroid_r += [(np.ceil(np.nanmean(label_centroids[:,0])), np.ceil(np.nanmean(label_centroids[:,1])))]             
    uber_centroids = uber_centroid_l + uber_centroid_r # list of tuples giving indices for the centroid of each DBSCAN cluster
            
            #img = (catmask_img==cat) # build category mask
#            for i in range(4): #iterate R, G, B, A
#                if i == 3: #4th dimension is luminance (A)
#                    #pdb.set_trace()
#                    all_cat_img[n,:,i] = proj_corr_lin
#                else: # set RGB
#                    color = tableau20_colors(n)
#                    all_cat_img[n,img,i] = color[i]/255.
                                            

    #if SAVE_PLOT:
        #figpath1 = os.path.join(gmm_path,filename_short+'thresh_centroids_'+timestamp_str[:10])
        #fig1.savefig(figpath1,  dpi=300, transparent=True)
        #figpath2 = os.path.join(gmm_path,filename_short+'thresh_centroids_xy_'+timestamp_str[:10])
        #fig2.savefig(figpath2,  dpi=300, transparent=True)
        #figpath3 = os.path.join(gmm_path,filename_short+'projected_thresholds_'+timestamp_str[:10])
        #fig3.savefig(figpath3,  dpi=300, transparent=True)
        #figpath4 = os.path.join(gmm_path,filename_short+'clustered_centroids_'+timestamp_str[:10])
        #fig4.savefig(figpath4,  dpi=300, transparent=True)    

    return submask_l, submask_r, l_db_cores, r_db_cores, uber_centroids, all_label_map

def get_centroids_from_seeds(dat, hemimask, thresh):    
    mask_img = dat*hemimask
    cy, cx, idx, corrmask = get_centroid(mask_img, thresh) # corrmask useful to debug correctness of centroid detection
    return cy, cx, idx
            
def get_centroid(img, thresh):
    idx_x = []
    idx_y = []
    corrmask = np.array(img>thresh) # do threshold
    idx_x, idx_y = np.nonzero(img>thresh) # get all indices of non-thresholded pixels
    corrmask = np.ones(mask.shape, dtype='int32')*corrmask #turn boolean mask into integers
    centroid_x = np.nanmean(idx_x)
    centroid_x_idx = np.ceil(centroid_x).astype('int16')
    centroid_y = np.nanmean(idx_y)
    centroid_y_idx = np.ceil(centroid_y).astype('int16')
    centroid_idx_out = (centroid_x_idx, centroid_y_idx)
    return centroid_y, centroid_x, centroid_idx_out, corrmask

def pca_ica(dat, num_ica):
     ica = FastICA(n_components=num_ica)
     ica_components = ica.fit_transform(all_corr_pca)
     ica_img = np.zeros([masky*maskx, num_ica])
     ica_img[:] = np.nan
     ica_img[pushmask,:] = ica_components
     ica_img = ica_img.T
     ica_img = np.reshape(ica_img, [num_ica, masky, maskx])
     return ica_img

def canny_edge(proj_threshcorr, corr_thresh):
    #assert proj_threshcorr.shape[0] == len(corr_thresh)
    proj_threshcorr[np.isnan(proj_threshcorr)] = 0 # zap nans
    edges = np.zeros([len(corr_thresh), masky, maskx])
    fig2 = plt.figure(figsize = (12,4))
    fig2.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=.05, wspace=.05)
    for n in range(len(corr_thresh)): 
        maxdat = np.max(proj_threshcorr[n,:,:])
        threshhigh = maxdat/10
        threshlow = threshhigh/2
        edges[n,:,:] = feature.canny(proj_threshcorr[n,:,:], 3, threshlow, threshhigh)
        ax2 = fig2.add_subplot(2, 5, n+1, xticks=[], yticks=[])   
        ax2.imshow(edges[n,:,:], cmap='gray_r')    
    if SAVE_PLOT:
        figpath = os.path.join(gmm_path,filename_short+'thresh_corr_borders_'+timestamp_str[:10])
        plt.savefig(figpath,  dpi=300, transparent=True)
    return edges
    
def dbscan_centroids(dat):
    db_model = DBSCAN(eps=4, min_samples=100).fit(dat)
    db_cores = db_model.core_sample_indices_
    centroid_labels = db_model.labels_
    centroid_cores = dat[db_cores,:]
    return centroid_cores, db_cores, centroid_labels

def gmm_centroid_map(pushmask, predictions, n_clusters, min_thresh, best_k):
    #pdb.set_trace()
    fig = plt.figure(figsize=(4, 4))
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=.05, wspace=.05)
    ax = fig.add_subplot(1, 1, 1)   
    dat = np.zeros([masky*maskx])
    dat[:] = np.nan
    dat[pushmask] = predictions
    dat = dat.reshape([masky, maskx])
    #cmap_colors = [rgb_to_hex(tableau20_colors(k)) for k in range(n_clusters)]
    #cmap = colors.ListedColormap(cmap_colors)
    #bounds = np.arange(n_clusters)
    #norm = colors.BoundaryNorm(bounds, cmap.N)
    #ax.imshow(dat, interpolation='nearest', cmap=cmap, norm=norm)
    ax.imshow(dat, interpolation='nearest', cmap='gnuplot2')
    
    if SAVE_PLOT:
        special = '_k=' + str(best_k) + '_' + gmm_type + '_corr_thresh=' + str(min_thresh) + '_bestk=' + str(n_clusters)
        figpath = os.path.join(gmm_path,filename_short+'_gmm_centroidmap_'+special+timestamp_str[:10])
        plt.savefig(figpath,  dpi=300, transparent=True)
        plt.close()

def rgb_to_hex(rgb):
    return '#%02x%02x%02x' % rgb
    
def euclidean_cost_centroids(img, thresh):
    #pdb.set_trace()
    fig6,ax6 = plt.subplots(figsize = (4,4))    
    mean_diff = np.empty(len(thresh))
    its = 100 # number of times to randomly calculate distances
    for n,t in enumerate(thresh[:-1]):
        dist0 = coord_dist(img[n,:,:], img[n-1,:,:])
        dist1 = coord_dist(img[n,:,:], img[n+1,:,:])
        ridx = np.random.randint(0, img.shape[1], (img.shape[1], its))
        rand_dist = [np.mean(coord_dist(img[n,:,:], img[n,ridx[:,k],:])) for k in range(its)]
        diff = np.abs(dist0) + np.abs(dist1) / np.mean(rand_dist)
        mean_diff[n] = np.mean(diff)  
    ax6.plot(thresh[1:-1], mean_diff[1:-1])
    ax6.set_xlabel('threshold')
    ax6.set_ylabel('cost')
    ax6.set_title('Correlation threshold cost function')
    if SAVE_PLOT:
        figpath = os.path.join(gmm_path,filename_short+'_edge_stability_'+timestamp_str[:10])
        plt.savefig(figpath,  dpi=300, transparent=True)
    return mean_diff, rand_dist

def coord_dist(coords1, coords2):
    dist = np.sqrt((coords1[:,0] - coords2[:,0])**2 + (coords1[:,1] - coords2[:,1])**2)
    return dist
    
def thresh_corr(all_corr, maskpathleft, maskpathright, corr_thresh):
    maskleft = tiff.imread(maskpathleft) 
    maskleft = maskleft/255
    maskright = tiff.imread(maskpathright)
    maskright = maskright/255    
    all_centroids_l, all_centroids_r, submask_l, submask_r = thresh_corr_centroids(
                                                                all_corr, 
                                                                maskleft, 
                                                                maskright, 
                                                                corr_thresh)  
    
    all_submask = submask_l + submask_r                                               
    all_centroids = np.concatenate((all_centroids_l, all_centroids_r), axis=1)
    if len(corr_thresh)>1:
        cost, rand_dist = euclidean_cost_centroids(all_centroids, corr_thresh)
    #min_thresh = np.nonzero(cost == np.min(cost[1:-1]))[0].astype('i2') # index of optiminum threshold
    #opt_thresh = corr_thresh[min_thresh] # the actual value of optimum threshold
    opt_thresh = [.75]
    min_thresh = np.nonzero(corr_thresh == opt_thresh)[0]
    opt_centroids = np.squeeze(all_centroids[min_thresh,:,:])
    
    #pdb.set_trace()
    submask_l, submask_r, l_db_cores, r_db_cores, uber_centroids, all_label_map = cluster_using_DBSCAN(
                                                                np.squeeze(all_centroids_l[min_thresh,:,:]), 
                                                                np.squeeze(all_centroids_r[min_thresh,:,:]), 
                                                                maskleft, 
                                                                maskright, 
                                                                min_thresh)
    
    fig5 = plt.figure(figsize = (12,4))    
    fig5.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=.05, wspace=.05)
    uber_masks = np.zeros([len(uber_centroids), masky, maskx])
    all_corr_mov = all_corr_to_image(all_corr)    
    for ii, idx in enumerate(uber_centroids):
        pushmask_idx = masky*idx[0]+idx[1]
        lin_idx = [nn for nn in range(pushmask.shape[0]) if pushmask[nn] == pushmask_idx]
        uber_masks[ii,:,:] = all_corr_mov[lin_idx,:,:]    
        uber_masks[uber_masks<opt_thresh] = np.nan        
        ax5 = fig5.add_subplot(3, 5, ii, xticks=[], yticks=[])
        ax5.scatter(uber_centroids[ii][1], uber_centroids[ii][0], color='blue', edgecolors='white')
        ax5.imshow(mask, vmin = 0, vmax = 2, cmap = plt.get_cmap('gray_r'))
        ax5.imshow(uber_masks[ii,:,:], cmap='gnuplot2')

    #pdb.set_trace()
    all_label_map = all_label_map.reshape([all_label_map.shape[0], masky, maskx])
    fig6 = plt.figure(figsize = (12,4))
    fig6.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=.05, wspace=.05)
    for ii in range(all_label_map.shape[0]):
        ax6 = fig6.add_subplot(3, 5, ii, xticks=[], yticks=[])
        ax6.scatter(uber_centroids[ii][1], uber_centroids[ii][0], color='blue', edgecolors='white')
        ax6.imshow(mask, vmin = 0, vmax = 2, cmap = plt.get_cmap('gray_r'))
        ax6.imshow(all_label_map[ii,:,:], vmin = 0, vmax = 1, cmap = plt.get_cmap('gray_r'))        
        
    fig7 = plt.figure(figsize = (4,4)) 
    fig7.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=.05, wspace=.05)
    ax7 = fig7.add_subplot(1, 1, 1, xticks=[], yticks=[])
    ax7.imshow(np.nanmax(uber_masks, axis=0), cmap='gnuplot2')
    ax7.scatter(zip(*uber_centroids)[1], zip(*uber_centroids)[0], color='blue', edgecolors='white')
    
    return opt_centroids, all_submask, uber_centroids, uber_masks, all_label_map
#  all_corr_to_image np.array([2,3,4,5,6,7,8,9,10,11,12,13,15,17,19,21,24,26,30,50])
#    #kays = np.array([8])
#    predictions = np.zeros([kays.shape[0], opt_centroids.shape[0]])
#    bics = np.zeros(kays.shape[0])
#    for k, kay in enumerate(kays):
#        gmm = build_gmm(kay)
#        cent_aic_mean, cent_bic_mean, cent_prob_samples, cent_respos, cent_datapoints = train_test_gmm(gmm, opt_centroids, kay)
#        predictions[k,:] = gmm.predict(opt_centroids)  
#        bics[k] = gmm.bic(opt_centroids)  
#        #cent_aic_mean, cent_bic_mean, cent_prob_samples, cent_respos, cent_datapoints = train_test_gmm(gmm, opt_centroids, kay)
#        #predictions_l = gmm.predict(opt_centroids_l)
#
#    min_bic_idx = bics.argmin()
#    best_k = kays[min_bic_idx]
#    print 'bic minimizes at k = ' + str(best_k) + ' clusters'
#    
#    fig,ax = plt.subplots(figsize=(4, 4))  
#    ax.plot(kays, bics)
#    ax.legend(['minimum bic at ' + str(best_k) + ' clusters'])
#    ax.set_xlabel('cluster number (k)')
#    ax.set_ylabel('BIC')
#    ax.set_title('BIC minimization')
#    
#    maskspace_predictions = np.zeros([all_corr.shape[0]]) 
#    maskspace_predictions[:] = np.nan
#    maskspace_predictions[all_submask] = np.squeeze(predictions[min_bic_idx, :])
    #maskspace_predictions[submask_l] = predictions_l # if submasks have overlap (they might depending on how accurently teh submasks are drawn), the left will overwrite the right
    
#    if SAVE_PLOT:
#        special = '_k=' + str(k) + '_' + gmm_type + '_corr_thresh=' + str(min_thresh)
#        figpath = os.path.join(gmm_path,filename_short+'_centroid_gmm_bics_'+special+timestamp_str[:10])
#        plt.savefig(figpath,  dpi=300, transparent=True)
#        plt.close()
#        
#    gmm_centroid_map(pushmask, maskspace_predictions, kays[min_bic_idx], min_thresh, best_k)
#    return best_k
    
    
############# load data and mask

mask = tiff.imread(maskpath) # mask must be 8bit thresholded as 0 & 255
mask = mask/255
masky = mask.shape[0]
maskx = mask.shape[1]
mask_idx, pullmask, pushmask = mask_to_index(mask)

datalength = [pushmask.shape[0],masky,maskx]

try: # if "all_corr" is in memory already
    all_corr
except NameError:
    if hdf5_format:
        f = h5py.File(filepath, 'r')
        imported_data = f['all_corr']
        all_corr = np.copy(imported_data)
        #f.close()
    else:            
        all_corr = np.load(filepath, 'r') #load numpy array 

#if rsamp_factor != 1:
#    all_corr = rand_tsamp(all_corr, rsamp_factor, masky, maskx, pushmask)


############# normalize correlation matrix to the correlation psf

if do_psf:    
    all_corr_mov = all_corr_to_image(all_corr)    
    psf = correlation_psf(all_corr_mov, mask_idx, psf_width)
    
    #print 'deconvolving...'
    all_corr_psfnorm = np.zeros([all_corr_mov.shape[0], masky, maskx])
    #for n in range(all_corr_mov.shape[0]):
        #all_corr_psfnorm[n,:,:] = all_corr_mov[n,:,:] - psf[n,:,:]
      
    plt.plot(psf[:,psf_width])
    #del all_corr_mov
    #figpath_psf = os.path.join(corr_path,filename_short+'psf_image_')
    #plt.savefig(figpath_psf,  dpi=300, transparent=True)
    #plt.close()

############# analyze thresholded correlation matrix

if do_thresh_corr:    
    #corr_thresh = [.3,.4,.5,.55,.6,.65,.7,.75,.8,.85,.9,.95,.99]
    corr_thresh = [.75]
    opt_centroids, all_submask, uber_centroids, uber_masks, all_label_map = thresh_corr(all_corr, maskpathleft, maskpathright, corr_thresh)
    
    
############# do PCA and GMM

if scan_pcas:    
    pcs = np.array([20])
    #pcs = np.array([15,20])
    kays = np.array([20,50,100])    
    #kays = np.array([2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,30,40,50,100,500])
    #kays = np.array([4,10,20])
    #rsamps = np.array([1])
    
    #iter_dim = rsamps
    iter_dim = pcs #set dimension to iterate over here
    aic_mean = np.zeros([iter_dim.shape[0], kays.shape[0]])
    bic_mean = np.zeros([iter_dim.shape[0], kays.shape[0]])
    mean_residual = np.zeros([iter_dim.shape[0], kays.shape[0], masky, maskx])
    
    for x in range(iter_dim.shape[0]): # for each size of PCA space
    
        pc_end = iter_dim[x]
        #rsamp_factor = rsamps[x]
        rsamp_factor = 1
        #pc_end = 50
#        if rsamp_factor != 1:
#            all_corr_ridx, ridx = rand_samp(all_corr, rsamp_factor)
#        else:
#            all_corr_ridx = all_corr #create new copy even if no rsamp used
        eigenmaps_img, all_corr_pca, variances = pca(all_corr, pc_start, pc_end)       
        
        for kk in range(kays.shape[0]): # for each number of clusters
        
            k = kays[kk] # overwrite global k
            if do_dpgmm:
                gmm = build_dpgmm(k)
            else:
                gmm = build_gmm(k)

            aic_mean[x,kk], bic_mean[x,kk], all_prob_samples, responsibilities, num_datapoints = train_test_gmm(gmm, all_corr_pca, k)
            
            if do_pcaica:
                ica_img = pca_ica(all_corr_pca, k)
                plot_all_clusters(ica_img)
                
            # do plotting            
            plot_predict(all_corr_pca, k, pc_end, rsamp_factor)
            plot_logprob(all_prob_samples, k, pc_end, rsamp_factor)
            get_DPP(pc_start, pc_end, k, num_datapoints)
            mean_residual[x,kk,:,:] = plot_mean_respos(responsibilities)
    
    special = '_'+gmm_type+'_gmm_scanK_'+str(kays[0])+'to'+str(kays[-1])+'_scanPCs_'+str(pcs[0])+'to'+str(pcs[-1]) # special string to add to filename        
    bicpath = os.path.join(gmm_path,filename_short+'bic_'+special+'scan_k_scan_rsamp'+timestamp_str[:10]+'.txt') 
    aicpath = os.path.join(gmm_path,filename_short+'aic_'+special+'scan_k_scan_rsamp'+timestamp_str[:10]+'.txt')
    residualpath = os.path.join(gmm_path,filename_short+'residual_'+special+'scan_k_scan_rsamp'+timestamp_str[:10]+'.txt')
    np.savetxt(bicpath, bic_mean, delimiter=',')  
    np.savetxt(aicpath, aic_mean, delimiter=',')
    np.savetxt(residualpath, mean_residual, delimiter=',')

    
if single_pca:    # just build GMM with one PCA space and one k
    
    pc_end = 20
    k=20
    
    eigenmaps_img, all_corr_pca, variances = pca(all_corr, pc_start, pc_end)
    #del all_corr
    pca_components = pc_end-pc_start # make pca_components global
    if do_pcaica:
         ica = FastICA(n_components=num_ica)
         ica_components = ica.fit_transform(all_corr_pca)
         ica_img = np.zeros([masky*maskx, num_ica])
         ica_img[:] = np.nan
         ica_img[pushmask,:] = ica_components
         ica_img = ica_img.T
         ica_img = np.reshape(ica_img, [num_ica, masky, maskx])
         plot_all_clusters(ica_img)
    
    gmm = build_gmm(k) 
    aic_mean, bic_mean, all_prob_samples, responsibilities, num_datapoints = train_test_gmm(gmm, all_corr_pca, k)  
    get_DPP(pc_start, pc_end, k, num_datapoints)
    
    all_predict, img_predict = predict_class(all_corr_pca)
    kmask, kmask_img = predict_masks(all_predict, k)
    plot_predict(all_corr_pca, k, pc_end, rsamp_factor)
    plot_logprob(all_prob_samples, k, pc_end, rsamp_factor)
    plot_all_masks(kmask_img)
    #mean_residual = plot_mean_respos(responsibilities)
    
    print 'BIC = ', bic_mean   
    print 'AIC = ', aic_mean
    #print 'Calinski Harabasz criterion = ', ch