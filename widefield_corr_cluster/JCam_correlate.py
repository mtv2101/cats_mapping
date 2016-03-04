#!/shared/utils.x86_64/python-2.7/bin/python

"""
JCam_correlate_cluster:

1. Load a movie and a binary mask of values 0,255
2. Downsample movie in x/y
3. Use per-pixel gaussian filter as f0 to calculate df/f
4. Bandpass filter timeseries
5. Do spatial correlation within mask pixels using random subset of movie frames

Created on Fri Dec 10 11:21:11 2014

@author: mattv
"""


# In[1]:

import os, sys
import platform
import datetime
import numpy as np
import h5py
import scipy.signal as sig
import timeit
if platform.system() == 'Linux':
    import matplotlib
    matplotlib.use('Agg')
if platform.system() == 'Linux': 
    sys.path.append(r'/data/nc-ophys/Matt/aibs/') # if windows local machine this should already be in base path    
    import CorticalMapping.tifffile as tiff # need this to view df/f movieexit
else:
    import aibs.CorticalMapping.tifffile as tiff



'''  '''
'''  '''   
'''  '''
   
   
   
   
class Jcorr(object): # correlate centroid clusters
    
    def __init__(self,
                 filepath,
                 
                 
                 
### Set filepaths and create directory structure

#if platform.system() == 'Windows':
#    #basepath = r'\\aibsdata2\nc-ophys\Matt\corrdata\150225_M166910'
#    basepath = r'\\aibsdata2\nc-ophys\CorticalMapping\IntrinsicImageData\150730-M187201'   
#    #basepath = r'\\aibsdata2\nc-ophys\Matt\Sparktest'    
#    
#elif platform.system() == 'Linux':  
#    #basepath = r'/data/nc-ophys/Matt/corrdata/150225_M166910/'
#    basepath = r'/data/nc-ophys/CorticalMapping/IntrinsicImageData/150804-M187474/'

basepath, filename = os.path.split(filepath)

corr_path = os.path.join(basepath, 'correlations')

# choose to make directory structure if it doesn't already exist
if os.path.isdir(corr_path) == False:
    os.mkdir(corr_path)

# enter file names
#filename = '010101JCamF105_2_2_1.npy' #file to open
filename_short = filename[:-4] # cut off '.npy'
#filepath = os.path.join(basepath, filename)
timestamp_str = str(datetime.datetime.now()) # this may be added to save strings later

if platform.system() == 'Windows':
    tif_ext = '.tif'
elif platform.system() == 'Unix' or platform.system() == 'Darwin':
    tif_ext = '.tiff'

# In[3]:
### set things to do and save

mask = False
dff = True # if false do df
SAVE_DFF = False
SAVE_F0 = False
old_format = False 
do_corr = True
epoch = False #[300000, 360000] # if epoch, set tuple as range of frames
img_source = 'npy' #'mapping_tif' #orca_flash
do_bandpass = False

# GLOBALS
size = 256 # size of square image in pixels
rsamp_factor = 100 # use random time samples for correlation, makes vectorlength smaller by this factor.  1=no downsampling
dsfactor = 2 #downsample Jcam in space by this factor.  Mask size must match originalsize/dsfactor
frameNum = -1 # load this number of frames - will error if longer than source
window = 60 # length of boxcar filter in seconds, used to calculate f0
lag = 0 # cross-correlation with time lag of this many samples

if mask:
   maskname = 'mask_256.tif' # mask size should match the final size of the correlation matrix
   maskpath = os.path.join(basepath,maskname) 

print 'rsamp_factor = ' + str(rsamp_factor)
print 'dsfactor = ' + str(dsfactor)
print 'frameNum = ' + str(frameNum)
print 'window = ' + str(window)
print 'lag = ' + str(lag)
print 'SAVE_DFF = ' + str(SAVE_DFF)
print 'old_format = ' + str(old_format)
print 'do_corr = ' + str(do_corr)
print 'img_source = ' + str(img_source)

# analysis paramater string to add to saved filenames
filenamenote = '_tx' + str(rsamp_factor) + '_sx' + str(dsfactor) + '_lag_' + str(lag) + '_dff_' + str(dff) # tx is time downsample, sx is space downsample
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
        exposureTime = 10.
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
        print 'please choose a window smaller than half the length of time you are analyzing'
    # pre-allocate matrices
    kernal = sig.gaussian(win, win/8)
    padsize = mov_tiny.shape[0] + (win*2) - 1
    mov_pad = np.zeros([padsize], dtype=('float'))
    #mov_ave = np.zeros([padsize+win-1], dtype=('float'))
    mov_dff = np.zeros([mov_tiny.shape[0], mov_tiny.shape[1], mov_tiny.shape[2]]) 
    delta_map = np.zeros([mov_tiny.shape[1], mov_tiny.shape[2]])
    # baseline subtract by gaussian convolution along time (dim=0)   
    yidx = np.array(mask_idx[0])
    xidx = np.array(mask_idx[1])
    print 'per-pixel baseline subtraction using gaussian convolution ...'
    #pdb.set_trace()
    for n in range(len_iter):
        # put data in padded frame
        mov_pad = pad_vector(mov_tiny[:,yidx[n],xidx[n]], win)
        # moving average by convolution
        mov_ave = sig.fftconvolve(mov_pad, kernal/kernal.sum(), mode='valid')
        # cut off pad
        mov_ave = mov_ave[win/2:(-win/2)-1]
        if n == 0:
            mov_ave_accum = np.zeros(mov_ave.shape[0])
        else:
            mov_ave_accum = ((mov_ave_accum*(n-1))+mov_ave)/n 
        delta_map[yidx[n],xidx[n]] = np.mean(mov_ave[:win])-np.mean(mov_ave[-win:])
        # and now use moving average as f0 for df/f
        if dff:
            mov_dff[:,yidx[n],xidx[n]] = (mov_tiny[:,yidx[n],xidx[n]] - mov_ave)/mov_ave
        else:
            mov_dff[:,yidx[n],xidx[n]] = (mov_tiny[:,yidx[n],xidx[n]] - mov_ave)
            
    return mov_dff, mov_ave_accum, delta_map

def pad_vector(dat, win):
    tlen = dat.shape[0]      
    pad_start = dat[0:win]+(dat[0]-dat[win])
    pad_end = dat[tlen-win:]+(dat[-1]-dat[tlen-win])
    dat_pad = np.append(np.append(pad_start, dat), pad_end)
    return dat_pad
    
def corr_map(sliced_mov_dff, corr_constant, seed_idx, lag):
    seed_pixel_ts = get_seed_pixel_ts(lag, sliced_mov_dff, seed_idx)
    # calc correlation map
    cor = (add_dims(seed_pixel_ts,sliced_mov_dff)*sliced_mov_dff).sum(axis=0)
    cor = cor/corr_constant
    cor = np.squeeze(cor/np.sqrt(np.dot(seed_pixel_ts,seed_pixel_ts)))
    print seed_idx
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
    
def rand_tsamp(mov, rsamp_factor):
    r_subsamp = np.ceil(mov.shape[0]/rsamp_factor)
    r_subsamp = r_subsamp.astype('i32')
    ridx = np.random.randint(0, mov.shape[0], r_subsamp)
    mov = mov[ridx,:,:]
    return mov
    
def get_seed_pixel_ts(lag, sliced_mov_dff, seed_idx): # 
    if lag < 0: 
        lag_abs = np.abs(lag)
        seed_pixel_ts = np.pad(sliced_mov_dff[:-lag_abs, seed_idx], [lag_abs, 0], mode='median')
    elif lag > 0:
        seed_pixel_ts = np.pad(sliced_mov_dff[lag:, seed_idx], [0, lag], mode='median')
    elif lag == 0:
        seed_pixel_ts = sliced_mov_dff[:, seed_idx]
    return seed_pixel_ts

def FIR_bandpass(lowcut, highcut, fs, filt_length):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    a = sig.firwin(filt_length, low, window='blackmanharris')
    b = - sig.firwin(filt_length, high, window = 'blackmanharris'); 
    b[filt_length/2] = b[filt_length/2] + 1
    d = - (a+b); # combine lowpass and highpass into bandpass
    d[filt_length/2] = d[filt_length/2] + 1
    return d
    
def FIR_lowpass(cut, fs, filt_length):
    nyq = 0.5 * fs
    corner = cut / nyq
    h = sig.firwin(filt_length, corner, window='hamming')
    return h
    
def do_correlation(mov_dff):
    
    mov_dff = mov_dff.reshape([mov_dff.shape[0], mov_dff.shape[1]*mov_dff.shape[2]])  
    
    sliced_mov_dff = mov_dff[:,pullmask] # pull just the pixels within the mask
    del mov_dff
    
    start_time = timeit.default_timer()
    
    all_corr = np.zeros([len_iter, len_iter], dtype='float')
    corr_constant = np.sqrt(np.sum(sliced_mov_dff*sliced_mov_dff,axis=0), dtype='float')        
    all_corr = [corr_map(sliced_mov_dff, corr_constant, n, lag) for n in range(len_iter)] # do the correlation
    all_corr = np.array(all_corr) #convert from list of arrays to 2-d numpy array
    
    corr_time = timeit.default_timer() - start_time
    print 'correlation took ' + str(corr_time) + ' seconds'
    
    return all_corr
    
def save_corr(all_corr, path):
    output = h5py.File(path, 'w')
    output.create_dataset('all_corr', data=all_corr)
    output.close()

def generate_mask(img, percentage):
    
    img_dims = img.shape
    bitdepth = 2**16
    img_hist, img_bins = np.histogram(img, bitdepth/100, [0,bitdepth])
    background_proportion = (512**2)/(100/percentage)
    cum = np.cumsum(img_hist)
    idx = cum[cum<background_proportion].shape[0]
    thresh = np.floor(img_bins[idx]).astype('uint16')
    
    mask = np.zeros(img_dims[0]*img_dims[1])
    img_flat = img.reshape(img_dims[0]*img_dims[1])
    mask[img_flat>thresh] = 1
    mask = mask.reshape(img_dims)
    
    return mask
    
    
# In[4]:
### Load JCam movie and a mask to isolate the analysis region

if img_source == 'old_camera':
    mov, exposure = importRawJCam(filepath)
elif img_source == 'mapping_tif':
    mov, exposure = ImportMappingTif(filepath)
elif img_source == 'orca_flash':
    mov, exposure = importMov(filepath, frameNum) # set frameNum = -1 to load whole file
elif img_source == 'npy':
    mov = np.load(filepath, 'r')
    exposure = 10.0 #in ms
    if frameNum != -1:        
        mov = mov[0:frameNum,:,:]
    if epoch:
        print 'analysing epoched data'
        epoch_idx = np.arange(epoch[0], epoch[1]) 
        mov = mov[epoch_idx,:] # pull just the pixels within the epoch 
    if rsamp_factor != 1:    
        mov = rand_tsamp(mov, rsamp_factor) 


# In[5]:
### Get indexing values from the mask

if mask:
    mask = tiff.imread(maskpath) # mask must be 8bit thresholded as 0 & 255
    mask = mask/255
else:
    mask = generate_mask(mov[0,:,:], 50) 
    
    
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

if do_bandpass:
    
    # first highpass using gaussian filter before doing bandpass
    mov_dff, mov_ave_accum, delta_map = moving_dff(mov_tiny, exposure, len_iter, mask_idx, window)
    all_corr = do_correlation(mov_dff)

    corrsavepath = os.path.join(corr_path,filename_short+'_cormap'+filenamenote+timestamp_str[:10])
    save_corr(all_corr, corrsavepath)
    if SAVE_DFF:
        dffsavepath = os.path.join(corr_path,filename_short+'_movie_'+timestamp_str[:10])
        np.save(dffsavepath, mov_dff)

    fs = 100.
    bands = [[.1, .3, 2048], # low pas, highpas in Hz, filter length in samples 
            [.3, 1, 1024], 
            [1, 3, 512], 
            [3, 10, 256], 
            [10, 30, 128]]  
    
    for nn, band in enumerate(bands):
        coeffs = FIR_bandpass(band[0], band[1], fs, band[2])
        for ii in range(mov_tiny.shape[1]):
            for jj in range(mov_tiny.shape[2]):
                mov_dff[:,ii,jj] = sig.filtfilt(coeffs, 1, mov_dff[:,ii,jj]) # use default pad values
                print 'filtering pixel #' + str(jj+mov_tiny.shape[1]*ii)
        
        bandpassnote = filenamenote + 'low' + str(band[0]) + '_high' + str(band[1]) + '_'        
        if SAVE_DFF:
            dffsavepath = os.path.join(corr_path,filename_short+'_movie_'+bandpassnote+timestamp_str[:10])
            np.save(dffsavepath, mov_dff)
        
        all_corr = do_correlation(mov_dff)
        
        corrsavepath = os.path.join(corr_path,filename_short+'_cormap'+bandpassnote+timestamp_str[:10])
        save_corr(all_corr, corrsavepath)
    
else: # do the vanilla df/f using average over the whole movie  
  
    mov_ave = np.mean(mov_tiny, axis=0)
    mov_dff = (mov_tiny - mov_ave)/mov_ave
    all_corr = do_correlation(mov_dff)
    
    corrsavepath = os.path.join(corr_path,filename_short+'_cormap'+timestamp_str[:10])
    save_corr(all_corr, corrsavepath)