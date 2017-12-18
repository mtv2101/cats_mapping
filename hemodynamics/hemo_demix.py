#!/shared/utils.x86_64/python-2.7/bin/python
"""
Created on Fri May 26 2017

@author: mattv
"""

# core
import os
import gc
import imp
import time
import timeit
import numpy as np

# analysis
import scipy.fftpack
import scipy.signal as sig
import scipy.ndimage as ndimage
from scipy.signal import butter, filtfilt
from skimage import transform
from skimage.measure import block_reduce



def butter_lowpass(sig, fs, corner, order=3):
    nyq = 0.5 * fs
    c = corner / nyq
    b,a = butter(order, c, btype='lowpass', analog=False)
    y1 = filtfilt(b, a, sig, axis=0)
    return y1

def butter_bandstop(sig, fs, corner1, corner2, order=3):
    nyq = 0.5 * fs
    c = [corner1 / nyq, corner2 / nyq]
    b,a = butter(order, c, btype='bandstop', analog=False)
    y1 = filtfilt(b, a, sig, axis=0)
    return y1

def butter_bandpass(sig, fs, corner1, corner2, order=3):
    nyq = 0.5 * fs
    c = [corner1 / nyq, corner2 / nyq]
    b,a = butter(order, c, btype='bandpass', analog=False)
    y1 = filtfilt(b, a, sig, axis=0)
    return y1

# estimate blood absorption of the two reflectance colors using Beer's Law
def beers_demixing(mov1_ds, mov2_ds, mov3_ds, path_470, path_530, path_575, path_640, smooth=False, corner=5.0, fs=100.0):
    
    ext_hbo_530 = 26450.0
    ext_hbr_530 = 32581.0
    ext_hbo_470 = 33209.0
    ext_hbr_470 = 16156.0
    ext_hbo_575 = 54424.0
    ext_hbr_575 = 40904.0
    ext_hbo_640 = 610.0
    ext_hbr_640 = 5149.0
    
    #start_time = timeit.default_timer()
    
    if smooth == True:
        mov1_ds = butter_lowpass(mov1_ds, fs, corner, order=5)
        mov2_ds = butter_lowpass(mov2_ds, fs, corner, order=5)
        mov3_ds = butter_lowpass(mov3_ds, fs, corner, order=5)
    
    mov1_f0 = np.mean(mov1_ds, axis=0)
    mov2_f0 = np.mean(mov2_ds, axis=0)
    mov3_f0 = np.mean(mov3_ds, axis=0)

    # get change in absorption coeffs from 530nm and 640nm reflectance(eq: 2.6):
    abs_575 = (-1/(path_575)) * np.log((mov2_ds)/mov2_f0)
    abs_640 = (-1/(path_640)) * np.log((mov3_ds)/mov3_f0)

    # estimate concentration hbo and hbr (eq: 2.7)
    hbr = ((ext_hbo_575*abs_640)-(ext_hbo_640*abs_575)) / \
        ((ext_hbo_575*ext_hbr_640)-(ext_hbo_640*ext_hbr_575))
        
    hbo = ((ext_hbr_575*abs_640)-(ext_hbr_640*abs_575)) / \
        ((ext_hbr_575*ext_hbo_640)-(ext_hbr_640*ext_hbo_575))

    # estimate absorption at 470nm GCaMP excitation and 530nm GCaMP emission (eq: 2.6)
    abs_470 = ext_hbo_470*hbo + ext_hbr_470*hbr
    abs_530 = ext_hbo_530*hbo + ext_hbr_530*hbr
    
    demixed = (mov1_ds/mov1_f0) * np.exp((abs_470*path_470)+(abs_530*path_530))
    
    #run_time = timeit.default_timer() - start_time
    #print 'demixing took ' + str(run_time) + ' seconds'
        
    return demixed*mov1_f0

def beers_demix(mov1, mov3, mov2):

    path_470 = 0.26
    path_530 = 0.27
    path_575 = 0.28
    path_640 = 3.85
    demixed_beers = beers_demixing(mov1, mov3, mov2, path_470, path_530, path_575, path_640, smooth=True)          

    return demixed_beers

def make_vasc_map(img, mask, percentile=95, size_thresh=1):
    img = img.astype('float')
    temp1 = (img - np.min(img))/((np.max(img)-np.min(img)))
    temp1[temp1>1.0] = 1.0
    temp2 = mask*gaussian_filter(temp1,size_thresh) 
    temp3 = (temp2/temp1)-1.0
    temp3[temp3 < 0.0] = 0.0
    temp = np.nanpercentile(temp3,percentile)
    temp3[temp3 > temp] = temp
    return temp3

def get_views(vasc_img, mov1_mask, mov2_mask, mov3_mask):

    X_0 = make_vasc_map(vasc_img, mask, percentile=95, size_thresh=2.0)
    X_0 = X_0.reshape([128*128])[pushmask]
    X_1 = np.nanmean(mov1_mask, axis=0)
    X_6 = np.nanstd(mov3_mask, axis=0)/np.nanmean(mov3_mask, axis=0)
    X_12 = np.nanstd(mov2_mask, axis=0)/np.nanmean(mov2_mask, axis=0)  
    X_14 = (1.0/mov1_mask.shape[0])*np.diag(np.dot(zscore(mov2_mask, axis=0).T, zscore(mov3_mask, axis=0)))

    X_offset = np.ones(len(pushmask))

    models = {'offset': X_offset, 'vasc': X_0, 'mean_470': X_1, \
              'std_575_norm': X_6, 'std_640_norm': X_12, 'corr575_640': X_14}

    #meta_views = np.array(models.values())
    
    return models

def meta_demix(mov1_mask, mov2_mask, mov3_mask, meta_coeffs):
    
    demixed = np.zeros(mov1_mask.shape)
    
    for x in range(mov1_mask.shape[1]):

        A = np.zeros([mov1_mask.shape[0], 3])
        A[:,0] = -np.log(mov1_mask[:,x])
        A[:,1] = -np.log(mov3_mask[:,x])
        A[:,2] = -np.log(mov2_mask[:,x])
        Abar = np.mean(A, axis=0)
        A = A - Abar
        Y = A[:,[1,2]]        
        sY = A[:,0]
        c = np.array([meta_coeffs[0,x],meta_coeffs[1,x]])
        sY_reg = Y.dot(c)
        demixed[:,x] = np.exp(-(sY-sY_reg.squeeze() + Abar[0]))
        
    return demixed
    
# the "meta-parameters" calculated from Cux2-Ai140 train mice
metas_cux2 = {'offset': [2.32903820e+00,  -1.40858596e+00],
              'vasc': [4.59051862e+00,  -2.89840140e+00],
              'mean_470': [-3.82666992e-05,   1.54242678e-05],
              'std_575_norm': [-4.24002980e+01,  -4.61664447e+00],
              'std_640_norm': [  9.79136904e-01,   8.82132149e+01],
              'corr575_640': [  5.78821154e-01,  -3.97188596e-01]}

# calculate "meta-views" of data - esentially projections of the data that can be used to predict regression maps  
meta_views = get_views(mov1[0,:,:], mov1_mask, mov2_mask, mov3_mask)

# get meta parameters that match to the meta views of the data
meta_params = [metas_cux2[model] for model in meta_views]
meta_params = np.array(meta_params)
meta_views = np.array(meta_views)

# estimate coefficient maps usign the meta views and meta parameters
meta_coeffs = np.dot(meta_params.T, meta_views)
            
#demix with meta params
demixed = meta_demix(mov1_mask, mov2_mask, mov3_mask, meta_coeffs)
#demix with Beer's Law
beers_demixed = beers_demix(mov1_mask, mov2_mask, mov3_mask)