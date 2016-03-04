#!/shared/utils.x86_64/python-2.7/bin/python

"""
Created on Thu Jun 04 10:56:23 2015

@author: mattv
"""

import platform
import os
import sys
import numpy as np
import h5py
from skimage import feature
import pickle
import datetime
if platform.system() == 'Linux': 
    sys.path.append(r'/data/nc-ophys/Matt/aibs/') # if windows local machine this should already be in base path 
    import matplotlib
    matplotlib.use('Agg')
import CorticalMapping.tifffile as tiff
import matplotlib.pyplot as plt


timestamp_str = str(datetime.datetime.now())

if platform.system() == 'Windows':
    basepath = r'\\aibsdata2\nc-ophys\Matt\corrdata\M177931\20150603'

elif platform.system() == 'Linux':  
    basepath = r'/data/nc-ophys/Matt/corrdata/M177929/20150505/'
    
filename = '20150603JCamF107deci1044_30000frames_cormap_tx1_sx2_lag_0_2015-06-08'
filename_short = '20150603JCamF107deci1044_30000frames_cormap_tx1_sx2'
maskname = 'mask256.tif'
mask_central = 'mask256_centralregion.tif'

corr_path = os.path.join(basepath, 'correlations')
gmm_path = os.path.join(basepath, 'gmm_cluster')
maskpath = os.path.join(basepath,maskname)
filepath = os.path.join(corr_path,filename)
mask_central_path = os.path.join(basepath,mask_central)

#GLOBALS
SAVE_PLOT = True
hdf5_format = True

def mask_to_index(mask):
    mask = mask.astype(np.intp) #64 bit integer
    mask_idx = np.nonzero(mask)
    pullmask = mask.reshape(mask.shape[0]*mask.shape[1]) #flatten image to vector
    pullmask = np.squeeze(np.array([pullmask==1])) #convert to boolean
    pushmask = np.nonzero(pullmask) # indexes for within the mask 
    pushmask = pushmask[0]
    return mask_idx, pullmask, pushmask

def canny_edge(threshcorr):
    threshcorr[np.isnan(threshcorr)] = 0 # zap nans
    maxdat = np.max(threshcorr)
    threshhigh = maxdat/10
    threshlow = threshhigh/2
    edges = feature.canny(threshcorr, 3, threshlow, threshhigh)  
    return edges

def get_edges(mov, thresh):   
    thresh_img = np.zeros([masky, maskx])
    thresh_img[:] = np.nan
    #thresh_img[(thresh+1)>mov>(thresh-1)] = 1
    thresh_img[mov>thresh] = 1
    out = canny_edge(thresh_img)
    return out
    
def all_corr_to_image(all_corr):
    n_samp = all_corr.shape[0]
    dat = np.zeros([n_samp, masky*maskx])
    dat[:,pushmask] = all_corr
    dat = dat.reshape([n_samp, masky, maskx])
    return dat

def get_masks(maskpath, mask_central_path):
    mask = tiff.imread(maskpath) # mask must be 8bit thresholded as 0 & 255
    mask = mask/255
    masky = mask.shape[0]
    maskx = mask.shape[1]
    mask_idx, pullmask, pushmask = mask_to_index(mask)
    mask_central = tiff.imread(mask_central_path) # mask must be 8bit thresholded as 0 & 255
    mask_central = mask_central/255
    return mask, mask_central, pullmask, pushmask, masky, maskx

def get_all_corr_mov(filepath):
    if hdf5_format:
        f = h5py.File(filepath, 'r')
        imported_data = f['all_corr']
        all_corr = np.copy(imported_data)
    else:            
        all_corr = np.load(filepath, 'r') #load numpy array 
    return all_corr
    
def edges_at_thresholds(all_corr_mov, thresh): 
    all_corr_canny_proj = np.zeros([len(thresh),masky,maskx])
    for i, threshold in enumerate(thresh):
        mov_edges = [get_edges(all_corr_mov[n,:,:], threshold) for n in range(all_corr_mov.shape[0])]
        proj_edges = np.nansum(mov_edges, axis=0)
        all_corr_canny_proj[i,:,:] = np.squeeze(proj_edges)
        del mov_edges
    return all_corr_canny_proj

def score_edges_differences(img, thresh):
    mean_diff = np.empty(len(thresh))
    for n,t in enumerate(thresh[:-1]):
        diff = np.abs(img[n,:,:] - img[n+1,:,:]) / np.mean(img[n,:,:])
        mean_diff[n] = np.mean(diff)    
    plt.plot(thresh[:-1], mean_diff[:-1])
    if SAVE_PLOT:
        figpath = os.path.join(gmm_path,filename_short+'_edge_stability_'+timestamp_str[:10])
        plt.savefig(figpath,  dpi=300, transparent=True)
    return mean_diff
    
def plot_edges(all_corr_canny_proj, thresh):
    fig3 = plt.figure(figsize = (12,4))    
    fig3.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=.05, wspace=.05)
    for n,t in enumerate(thresh):
        ax3 = fig3.add_subplot(3, 5, n+1, xticks=[], yticks=[])
        im = ax3.imshow(all_corr_canny_proj[n,:,:], cmap='gray_r', interpolation='nearest')
        im.set_clim(0,1500)
    if SAVE_PLOT:
        figpath = os.path.join(gmm_path,filename_short+'_allpredictions_'+timestamp_str[:10])
        plt.savefig(figpath,  dpi=300, transparent=True)
        #plt.close()
    
all_corr = get_all_corr_mov(filepath)

mask, mask_central, pullmask, pushmask, masky, maskx = get_masks(maskpath, mask_central_path)

all_corr_mov = all_corr_to_image(all_corr)

thresh = [.3,.4,.5,.55,.6,.65,.7,.75,.8,.85,.9,.95,.99]
    
all_corr_canny_proj = edges_at_thresholds(all_corr_mov, thresh)

all_corr_canny_central = all_corr_canny_proj*mask_central
all_corr_canny_central_lin = all_corr_canny_central.reshape([len(thresh),masky*maskx])

mean_diff = score_edges_differences(all_corr_canny_central, thresh)

plot_edges(all_corr_canny_proj, thresh)

picklepath3 = os.path.join(gmm_path,filename_short+'projected_edges.pkl')
pickle.dump(all_corr_canny_proj, open(picklepath3, "wb"))