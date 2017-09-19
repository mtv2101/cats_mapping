# -*- coding: utf-8 -*-
"""
Created on Tue Jun 02 17:40:06 2015

Load a correlation matrix and the binary mask of the brain used to create it

@author: mattv
"""

import platform
import os
import sys
import numpy as np
import h5py  
#import matplotlib.pyplot

if platform.system() == 'Windows':
    sys.path.append(r'\\aibsdata2\nc-ophys\Matt\aibs')
    basepath = r'\\aibsdata2\nc-ophys\CorticalMapping\IntrinsicImageData\150722-M177931'
    filename = '010101JCamF103_2_2_1_128x128_150722-M177931_20151103-144358_dec_cormap__tx100_sx1_nframes_17150_dff_True_20151103-144358_deci_02015-11-17'
    maskname = '010101JCamF103_2_2_1_128x128_150722-M177931_20151103-144358_deci_mask128.tif'
    
elif platform.system() == 'Linux' or 'Darwin':
    sys.path.append(r'/data/nc-ophys/Matt/aibs/') # if windows local machine this should already be in base path  
    basepath = r'/data/nc-ophys/Matt/corrdata/M183699/20150506/'
    filename = '20150505JCamF101deci244_cormap_tx1_sx2_lag_0_2015-05-16'
    maskname = 'mask256.tif'
    
# from the aibs package:
import CorticalMapping.tifffile as tiff # need this to view df/f movie

#corr_path = os.path.join(basepath, 'correlations')
corr_path = basepath
maskpath = os.path.join(basepath,maskname)
filepath = os.path.join(corr_path,filename)

def mask_to_index(mask):
    mask = mask.astype(np.intp) #64 bit integer
    mask_idx = np.nonzero(mask)
    pullmask = mask.reshape(mask.shape[0]*mask.shape[1]) #flatten image to vector
    pullmask = np.squeeze(np.array([pullmask==1])) #convert to boolean
    pushmask = np.nonzero(pullmask) # indexes for within the mask 
    pushmask = pushmask[0]
    return mask_idx, pullmask, pushmask
    
def maskspace_to_imagespace(dat, masky, maskx):
    dat_img = np.zeros([dat.shape[0], masky*maskx])
    dat_img[:,pushmask] = dat
    dat_img = dat_img.reshape([all_corr.shape[0], masky, maskx])
    return dat_img
    
# load the mask of the brain used in creating the correlation matrix
mask = tiff.imread(maskpath) # mask is 8bit thresholded as 0 & 255
mask = mask/255
masky = mask.shape[0]
maskx = mask.shape[1]
mask_idx, pullmask, pushmask = mask_to_index(mask)

#load the correlation matrix
f = h5py.File(filepath, 'r')
imported_data = f['all_corr']
all_corr = np.copy(imported_data)

#transform correlation matrix to image space
#output size is [seedpixels, y, x]
all_corr_img = maskspace_to_imagespace(all_corr, masky, maskx)

tiff.imshow(all_corr_img)