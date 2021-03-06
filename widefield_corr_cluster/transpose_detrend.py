#!/shared/utils.x86_64/python-2.7/bin/python

import numpy as np
import scipy.signal as sig
import tables as tb
import timeit
import pdb
import os
import sys
import platform
if platform.system() == 'Linux':
    sys.path.append(r'/data/nc-ophys/Matt/imagingbehavior')
    import aibs.CorticalMapping.tifffile as tiff
else:   
    import aibs.CorticalMapping.tifffile as tiff



def detrend(mov, mov_detrend, mask_idx, pushmask, frames, exposure, window, dff):    
    # define size of rolling average 
    expose = np.float(exposure/1000.) #convert exposure from ms to s
    win = np.float(window) # window in seconds
    win = win/expose
    win = np.ceil(win) 
    if win > len(frames)/2:
        print 'please choose a window smaller than half the length of time you are analyzing'
    # pre-allocate matrices
    kernal = sig.gaussian(win, win/8)
    padsize = len(frames) + (win*2) - 1
    mov_pad = np.zeros([padsize], dtype=('float32'))
    # load the data and transpose 
    mov_trans = mov[frames,:,:].reshape([len(frames), mov.shape[1]*mov.shape[2]]).T
    # baseline subtract by gaussian convolution along time (dim=0)   
    print 'per-pixel baseline subtraction using gaussian convolution ...\n'
    len_iter = pushmask.shape[0]
    print str(len_iter) + ' pixels'
    for n in range(len_iter):    
        # put data in padded frame
        mov_seg = np.squeeze(mov_trans[pushmask[n],:])
        mov_pad = pad_vector(mov_seg, win)
        # moving average by convolution
        mov_ave = sig.fftconvolve(mov_pad, kernal/kernal.sum(), mode='valid')
        # cut off pad
        mov_ave = mov_ave[win/2:(-win/2)-1]
        mov_ave = mov_ave.astype('float32')
        # and now use moving average as f0 for df/f
        if dff:
            mov_dff = (mov_seg - mov_ave)/mov_ave
        else:
            mov_dff = mov_seg - mov_ave 

        mov_detrend.append(mov_dff[None])   

def pad_vector(dat, win):
    tlen = dat.shape[0]      
    pad_start = dat[0:win]+(dat[0]-dat[win])
    pad_end = dat[tlen-win:]+(dat[-1]-dat[tlen-win])
    dat_pad = np.append(np.append(pad_start, dat), pad_end)
    return dat_pad

def generate_mask(mov, mask_savepath, percentage=50):  
    img = mov[0,:,:]
    img_dims = img.shape
    bitdepth = 2**16
    img_hist, img_bins = np.histogram(img, bitdepth/100, [0,bitdepth])
    background_proportion = (img_dims[0] * img_dims[1])/(100/percentage)
    cum = np.cumsum(img_hist)
    idx = cum[cum<background_proportion].shape[0]
    thresh = np.floor(img_bins[idx]).astype('uint16')    
    mask = np.zeros(img_dims[0]*img_dims[1])
    img_flat = img.reshape(img_dims[0]*img_dims[1])
    mask[img_flat>thresh] = 1
    mask = mask.reshape(img_dims)  
    
    mask_tosave = mask.astype('uint8')
    mask_tosave = mask_tosave*255 # for historical reasons, the mask is either 0 and 255
    tiff.imsave(mask_savepath, mask_tosave)
    
    return mask

def mask_to_index(mask):
    mask = mask.astype('uint16') #64 bit integer
    mask_idx = np.ndarray.nonzero(mask)
    pullmask = mask.reshape(mask.shape[0]*mask.shape[1]) #flatten image to vector
    pullmask = np.squeeze(np.array([pullmask==1])) #convert to boolean
    pushmask = np.ndarray.nonzero(pullmask) # indexes for within the mask 
    pushmask = pushmask[0]
    return mask_idx, pullmask, pushmask
    
def transpose_detrend(filepath, output_file, dff=True, start=0, stop=1000000, window=60):  
 
    start_time = timeit.default_timer()
    exposure = 10 # camera exposure in ms
    
    basepath, filename = os.path.split(filepath)
    mask_savepath = os.path.join(basepath, filename[:-3] + '_mask128.tif')

    # load data
    open_tb = tb.open_file(filepath, 'r')
    mov = open_tb.root.data

    frames = range(start, stop)
    if stop > mov.shape[0]:
        frames = range(start, mov.shape[0])
    print str(len(frames)) + ' frames will be detrended'
    
    tdim = mov[frames,:,:].shape[0]
    
    # setup masking variables
    mask = generate_mask(mov, mask_savepath=mask_savepath)
    mask_idx, pullmask, pushmask = mask_to_index(mask)
    
    # open output file which will be a space x time transposition of the movie
    fd = tb.openFile(output_file, 'w')
    filters = tb.Filters(complevel=1, complib='blosc')
    mov_detrend = fd.createEArray(fd.root, 
                    'data', 
                    tb.Float32Atom(), 
                    filters=filters,
                    expectedrows=pushmask.shape[0],
                    shape=(0, int(tdim)))
    
    # detrend the movie
    detrend(mov, mov_detrend, mask_idx, pushmask, frames, exposure, window, dff)
    detrend_time = timeit.default_timer() - start_time
    print 'detrending took ' + str(detrend_time) + ' seconds\n'
    
    fd.close()
    open_tb.close()
    
if __name__ == "__main__": 
    print ''