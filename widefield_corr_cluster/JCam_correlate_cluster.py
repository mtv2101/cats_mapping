#!/shared/utils.x86_64/python-2.7/bin/python

"""
JCam_correlate_cluster:

1. Load a movie and a binary mask of values 0,255
2. Downsample movie in x/y
3. Use boxcar average as f0 to calculate df/f
4. Do spatial correlation within mask pixels using random subset of movie frames

Created on Fri Dec 10 11:21:11 2014

@author: mattv
"""


# In[1]:

import os
import platform
import datetime
import sys
import time
import numpy as np
#import pickle
import scipy.signal as sig
if platform.system() == 'Linux':
    import matplotlib
    matplotlib.use('Agg')
if platform.system() == 'Linux': 
    sys.path.append(r'/data/nc-ophys/Matt/aibs/') # if windows local machine this should already be in base path    
import CorticalMapping.tifffile as tiff # need this to view df/f movie


# In[2]:
### Set filepaths and create directory structure

if platform.system() == 'Windows':
    #basepath = r'\\aibsdata2\nc-ophys\Matt\corrdata\150225_M166910'
    basepath = r'C:\Users\mattv\Desktop\local data\150225_M166910'   
    
elif platform.system() == 'Linux':  
    basepath = r'/data/nc-ophys/Matt/corrdata/150225_M166910/'
elif platform.system() =='Darwin':
    basepath = r'/data/nc-ophys/Matt/corrdata/150225_M166910/'

corr_path = os.path.join(basepath, 'correlations')
movie_path = os.path.join(basepath, 'corr_movies')
gmm_path = os.path.join(basepath, 'gmm_cluster')
pca_path = os.path.join(basepath, 'pca')

# choose to make directory structure if it doesn't already exist
#os.mkdir(corr_path)
#os.mkdir(movie_path)
#os.mkdir(gmm_path)
#os.mkdir(pca_path)

# enter file names
filename = '20150225JCamF102deci244' #file to open
filename_short = '20150225JCamF102deci244' #string to use in creating filenames
filepath = os.path.join(basepath, filename)
maskname = '102mask_64.tif' # mask size should match the final size of the correlation matrix
maskpath = os.path.join(basepath,maskname)
timestamp_str = str(datetime.datetime.now()) # this may be added to save strings later

if platform.system() == 'Windows':
    tif_ext = '.tif'
elif platform.system() == 'Unix' or platform.system() == 'Darwin':
    tif_ext = '.tiff'

# In[3]:
### set things to do and save

do_moving_dff = False
SAVE_DFF = False
old_format = False 
do_corr = True
img_source = 'orca_flash' #'mapping_tif' #orca_flash

# GLOBALS
rsamp_factor = 1 # use random time samples for correlation, makes vectorlength smaller by this factor.  1=no downsampling
dsfactor = 8 #downsample Jcam in space by this factor.  Mask size must match originalsize/dsfactor
frameNum = 1000 # load this number of frames - will error if longer than source
window = 30 # length of boxcar filter in seconds, used to calculate f0
lag = 0 # cross-correlation with time lag of this many samples

print 'rsamp_factor = ' + str(rsamp_factor)
print 'dsfactor = ' + str(dsfactor)
print 'frameNum = ' + str(frameNum)
print 'window = ' + str(window)
print 'lag = ' + str(lag)
print 'do_moving_dff = ' + str(do_moving_dff)
print 'SAVE_DFF = ' + str(SAVE_DFF)
print 'old_format = ' + str(old_format)
print 'do_corr = ' + str(do_corr)
print 'img_source = ' + str(img_source)

# analysis paramater string to add to saved filenames
filenamenote = '_tx' + str(rsamp_factor) + '_sx' + str(dsfactor) + '_lag_' + str(lag) + '_' # tx is time downsample, sx is space downsample
print filenamenote
       
# In[2]:
### Set defs

def ImportMappingTif(path): #used to import tiff stacks, most often with vasculature maps
    imageFile = tiff.imread(path)
    exposureTime = 10
    return imageFile, exposureTime
    
def importRawJCam(path,
                  dtype = np.dtype('>f'),
                  headerLength = 96, # length of the header, measured as the data type defined above               
                  columnNumIndex = 14, # index of number of rows in header
                  rowNumIndex = 15, # index of number of columns in header
                  frameNumIndex = 16, # index of number of frames in header
                  decimation = None, #decimation number
                  exposureTimeIndex = 17): # index of exposure time in header, exposure time is measured in ms
    '''
    import raw JCam files into np.array
        raw file format:
        data type: 32 bit sigle precision floating point number
        data format: big-endian single-precision float, high-byte-first motorola
        header length: 96 floating point number
        column number index: 14
        row number index: 15
        frame number index: 16
        exposure time index: 17
    '''
    openit = open(path, 'r+b')
    imageFile = np.fromfile(openit,dtype=dtype,count=-1)   
    openit.close()    
    columnNum = np.int(imageFile[columnNumIndex])
    rowNum = np.int(imageFile[rowNumIndex])    
    if decimation != None:
        columnNum = columnNum/decimation
        rowNum =rowNum/decimation    
    frameNum = np.int(imageFile[frameNumIndex])    
    if frameNum == 0: # if it is a single frame image
        frameNum = frameNum +1      
    exposureTime = np.float(imageFile[exposureTimeIndex])    
    imageFile = imageFile[headerLength:]    
    print 'width =', str(columnNum), 'pixels'
    print 'height =', str(rowNum), 'pixels'
    print 'length =', str(frameNum), 'frame(s)'
    print 'exposure time =', str(exposureTime), 'ms'    
    imageFile = imageFile.reshape((frameNum,rowNum,columnNum))     
    return imageFile, exposureTime
    
def importMov(path, frameNum): 
    if old_format: 
        dtype = np.dtype('>u2') #dtype for old JCam format
        columnNum = 512  
        rowNum = 512                           
        exposureTime = 20.
        bytenum = frameNum*rowNum*columnNum
    else:
        dtype = np.dtype('<u2') # < is little endian, 2 for 16bit
        columnNum = 512  
        rowNum = 512                           
        exposureTime = 20.
        bytenum = frameNum*rowNum*columnNum
    if frameNum == -1: #load all
        bytenum = -1 
    openit = open(path, 'r+b')
    imageFile = np.fromfile(openit,dtype=dtype,count=bytenum)  
    openit.close()
    frameNum = imageFile.shape[0]/(rowNum*columnNum)
    # in case number of loaded frames does not match frameNum, recalculate frameNum.  This is mostly for debugging purposes
    print 'width =', str(columnNum), 'pixels'
    print 'height =', str(rowNum), 'pixels'
    print 'length =', str(frameNum), 'frame(s)'
    print 'exposure time =', str(exposureTime), 'ms'    
    imageFile = imageFile.reshape((frameNum,rowNum,columnNum))
    return imageFile, exposureTime
    
def mask_to_index(mask):
    mask = mask.astype(np.intp) #64 bit integer
    mask_idx = np.ndarray.nonzero(mask)
    pullmask = mask.reshape(mask.shape[0]*mask.shape[1]) #flatten image to vector
    pullmask = np.squeeze(np.array([pullmask==1])) #convert to boolean
    pushmask = np.ndarray.nonzero(pullmask) # indexes for within the mask 
    pushmask = pushmask[0]
    return mask_idx, pullmask, pushmask
    
def sub_sample_movie(mov, dsfactor):
    print 'subsampling movie...'
    mov_tiny = np.zeros([mov.shape[0],mov.shape[1]/dsfactor, mov.shape[2]/dsfactor], dtype=('float'))
    for i in xrange(mov_tiny.shape[-2]):
        for j in xrange(mov_tiny.shape[-1]):
            mov_tiny[:,i,j] = mov[:, i*dsfactor:(i+1)*dsfactor,j*dsfactor:(j+1)*dsfactor].mean(-1).mean(-1)
    return mov_tiny

def moving_dff(mov_tiny, exposure, len_iter, mask_idx, window):    
    # define size of rolling average 
    exposure = np.float(exposure/1000) #convert exposure from ms to s
    window = np.float(window) # window in seconds
    win = window/exposure
    win = np.ceil(win) 
    if win > mov_tiny.shape[0]/2:
        print 'please choose a box-car window smaller than half the length of time you are analyzing'
    # pre-allocate matrices
    kernal = np.ones(win, dtype=('float'))
    padsize = mov_tiny.shape[0] + win*2
    mov_pad = np.zeros([padsize], dtype=('float'))
    mov_ave = np.zeros([padsize+win-1], dtype=('float'))
    mov_dff = np.zeros([mov_tiny.shape[0]-win, mov_tiny.shape[1], mov_tiny.shape[2]]) 
    # baseline subtract by per-pixel rolling average along time (dim=0)
    yidx = np.array(mask_idx[0])
    xidx = np.array(mask_idx[1])
    print 'dffing ...'
    for n in range(len_iter):
        # put data in padded frame
        mov_pad[win:(padsize-win)] = mov_tiny[:,yidx[n],xidx[n]]
        # moving average by convolution
        mov_ave = sig.fftconvolve(mov_pad, kernal)/win
        # cut off pad
        mov_ave = mov_ave[win*2:1+mov_ave.shape[0]-win*2]
        # and now use moving average as f0 for df/f
        mov_dff[:,yidx[n],xidx[n]] = (mov_tiny[(win/2):mov_tiny.shape[0]-(win/2),yidx[n],xidx[n]] - mov_ave)/mov_ave
    return mov_dff

    
def corr_map(seed_pixel_ts, sliced_mov_dff, corr_constant):
    # calc correlation map
    cor = (add_dims(seed_pixel_ts,sliced_mov_dff)*sliced_mov_dff).sum(axis=0)
    cor = cor/corr_constant
    cor = np.squeeze(cor/np.sqrt(np.dot(seed_pixel_ts,seed_pixel_ts)))
    return cor

def add_dims(seed_pixel_ts, sliced_mov_dff):
    for d in range(sliced_mov_dff.ndim-1):
        seed_pixel_ts = np.expand_dims(seed_pixel_ts,d+1)
    return seed_pixel_ts   
        
def smart_imshow_cscale(plot_probs):
    plot_probs_lin = np.reshape(plot_probs, [masky*maskx])
    img_median = np.nanmedian(plot_probs_lin)
    img_mad = np.nanmedian(np.absolute(plot_probs_lin - np.nanmedian(plot_probs_lin))) #mean absolute deviation
    upper_clim = img_median+(img_mad*3)
    lower_clim = img_median-(img_mad*3)
    return lower_clim, upper_clim
    
def rand_tsamp(mov_dff, rsamp_factor):
    r_subsamp = np.ceil(mov_dff.shape[0]/rsamp_factor)
    r_subsamp = r_subsamp.astype('i32')
    ridx = np.random.randint(0, mov_dff.shape[0], r_subsamp)
    mov_dff = mov_dff[ridx,:]
    return mov_dff

# In[4]:
### Load JCam movie and a mask to isolate the analysis region

if img_source == 'old_camera':
    mov, exposure = importRawJCam(filepath)
elif img_source == 'mapping_tif':
    mov, exposure = ImportMappingTif(filepath)
elif img_source == 'orca_flash':
    mov, exposure = importMov(filepath, frameNum) # set frameNum = -1 to load whole file


# In[5]:
### Get indexing values from the mask

mask = tiff.imread(maskpath) # mask must be 8bit thresholded as 0 & 255
mask = mask/255
    
masky = mask.shape[0]
maskx = mask.shape[1]
mask_idx, pullmask, pushmask = mask_to_index(mask)
#pullmask: boolean array defining mask
#pushmask: indexes that project masked data back to full image dimensions (linearlized)
len_iter = np.array([pushmask.shape]) #length of flattened mask - used for setting iterators


# In[]:
### Now pre-process movie to subtract background, and eliminate slow drift of the signal 
 
if dsfactor == 1:
    mov_tiny = mov
else:
    mov_tiny = sub_sample_movie(mov, dsfactor)
del mov

if do_moving_dff:
    #mov_dff, mov_ave = moving_dff(mov_tiny, exposure, len_iter, mask_idx, window)
    mov_dff = moving_dff(mov_tiny, exposure, len_iter, mask_idx, window)
else: # do the vanilla df/f using average over the whole movie    
    mov_ave = np.mean(mov_tiny, axis=0)
    mov_dff = (mov_tiny - mov_ave)/mov_ave
del mov_tiny 

if SAVE_DFF:
    dffsavepath = os.path.join(corr_path,filename_short+'_dffmovie_'+filenamenote+timestamp_str[:10])
    np.save(dffsavepath, mov_dff)
    
# In[]:
### pre-process df/f data for correlation

# flatten space dimensions of DF/F for correlation
mov_dff = mov_dff.reshape([mov_dff.shape[0], mov_dff.shape[1]*mov_dff.shape[2]])

# Randomly sub-sample time for correlation
if rsamp_factor != 1:    
    mov_dff = rand_tsamp(mov_dff, rsamp_factor)


# In[]:
### Correlate each pixel's df/f with every other pixel's df/f

if do_corr:
    all_corr = np.zeros([len_iter, len_iter], dtype='float')
    sliced_mov_dff = mov_dff[:,pullmask]
    corr_constant = np.sqrt(np.sum(sliced_mov_dff*sliced_mov_dff,axis=0), dtype='float')
    timer = time.time()
    print 'Doing correlation:'
    
    if rsamp_factor != 1 & lag != 0:
        print 'beware you are cross-correlating with random time samples creating an unknown time lag'
        
    for n in range(len_iter):  # for each seed pixel 
        
        if lag < 0: 
            lag_abs = np.abs(lag)
            seed_pixel_ts = np.pad(mov_dff[:-lag_abs, pushmask[n]], [lag_abs, 0], mode='median')
        elif lag > 0:
            seed_pixel_ts = np.pad(mov_dff[lag:, pushmask[n]], [0, lag], mode='median')
        elif lag == 0:
            seed_pixel_ts = mov_dff[:, pushmask[n]]
            
        all_corr[n,:] = corr_map(seed_pixel_ts, sliced_mov_dff, corr_constant)
    print '  Correlation time (s):', time.time() - timer

    
    corrsavepath = os.path.join(corr_path,filename_short+'_cormap'+filenamenote+timestamp_str[:10])
    np.save(corrsavepath, all_corr)
