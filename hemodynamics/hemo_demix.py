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
def est_hb(mov2_ds, mov3_ds, path_640, path_590_ref, smooth=False, corner=3.0, fs=100.0):

    ext_hbo_470 = 33209.2
    ext_hbr_470 = 16156.4
    ext_hbo_590 = 50000 
    ext_hbr_590 = 40000
    ext_hbo_640 = 442
    ext_hbr_640 = 4345.2
    ext_hbo_530 = 40000
    ext_hbr_530 = 40000
    
    
    start_time = timeit.default_timer()
    
    if smooth == True:
        mov2_ds = butter_lowpass(mov2_ds, fs, corner, order=5)
        mov3_ds = butter_lowpass(mov3_ds, fs, corner, order=5)
        
    mov2_f0 = np.mean(mov2_ds, axis=0)
    mov3_f0 = np.mean(mov3_ds, axis=0)

    # get change in absorption coeffs from 530nm and 640nm reflectance(eq: 2.6):
    abs_590 = (-1/(path_590_ref)) * np.log((mov2_ds)/mov2_f0)
    abs_640 = (-1/(path_640)) * np.log((mov3_ds)/mov3_f0)

    # estimate concentration hbo and hbr (eq: 2.7)
    hbr = ((ext_hbo_590*abs_640)-(ext_hbo_640*abs_590)) / \
        ((ext_hbo_590*ext_hbr_640)-(ext_hbo_640*ext_hbr_590))
        
    hbo = ((ext_hbr_590*abs_640)-(ext_hbr_640*abs_590)) / \
        ((ext_hbr_590*ext_hbo_640)-(ext_hbr_640*ext_hbo_590))

    # estimate absorption at 470nm GCaMP excitation and 530nm GCaMP emission (eq: 2.6)
    abs_470 = ext_hbo_470*hbo + ext_hbr_470*hbr
    abs_530 = ext_hbo_530*hbo + ext_hbr_530*hbr
    
    run_time = timeit.default_timer() - start_time
    print 'hbo/hbr estimation took ' + str(run_time) + ' seconds'
        
    return abs_470, abs_530

def correct_gcamp(mov1_ds, abs_470, abs_530, path_470, path_530):
    
    start_time = timeit.default_timer()
        
    mov1_f0 = np.mean(mov1_ds, axis=0)

    demixed = (mov1_ds/mov1_f0) * np.exp((abs_470*path_470)+(abs_530*path_530))
    
    run_time = timeit.default_timer() - start_time
    print 'demixing took ' + str(run_time) + ' seconds'
    return demixed

def demix(datapath, params, savepath):

    path_640 = 4.140704

    mov1_ds, mov2_ds, mov3_ds = load_data(datapath)

    abs470, abs530 = est_hb(mov2_ds, mov3_ds, path_640, params[2])

    demixed = correct_gcamp(mov1_ds, abs470, abs530, params[0], params[1])

    score_demixed = np.std(demixed, axis=0)  

    np.save(savepath, score_demixed)          

    return score_demixed