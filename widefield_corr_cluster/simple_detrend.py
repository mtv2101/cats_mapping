#!/shared/utils.x86_64/python-2.7/bin/python

import numpy as np
import scipy.signal as sig
import matplotlib.pyplot as plt 
import tables as tb
import timeit
import pdb



def detrend(mov, mov_detrend, mask_idx, pushmask, frames, exposure):    
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
    mov_trans = mov[:,:,:].reshape([mov.shape[1]*mov.shape[2], mov.shape[0]]).T
    #yidx = np.array(mask_idx[0])
    #xidx = np.array(mask_idx[1])
    # baseline subtract by gaussian convolution along time (dim=0)   
    print 'per-pixel baseline subtraction using gaussian convolution ...\n'
    len_iter = pushmask.shape[0]
    print str(len_iter) + ' pixels'
    for n in range(len_iter):    
        print n
        # put data in padded frame
        #pdb.set_trace()
        #mov_seg = mov[frames, yidx[n], xidx[n]]
        mov_seg = mov_trans[n,:]
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
        #pdb.set_trace()
        mov_detrend.append(mov_dff[None])   
    
    fd.close()


def pad_vector(dat, win):
    tlen = dat.shape[0]      
    pad_start = dat[0:win]+(dat[0]-dat[win])
    pad_end = dat[tlen-win:]+(dat[-1]-dat[tlen-win])
    dat_pad = np.append(np.append(pad_start, dat), pad_end)
    return dat_pad

def generate_mask(img, percentage):    
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
    return mask

def get_mask(mov):
	frame = mov[0,:,:] # mask using the first frame of the movie  
	mask = generate_mask(frame, 50)           
	plt.imshow(frame * mask)
	return mask

def mask_to_index(mask):
    mask = mask.astype('uint16') #64 bit integer
    mask_idx = np.ndarray.nonzero(mask)
    pullmask = mask.reshape(mask.shape[0]*mask.shape[1]) #flatten image to vector
    pullmask = np.squeeze(np.array([pullmask==1])) #convert to boolean
    pushmask = np.ndarray.nonzero(pullmask) # indexes for within the mask 
    pushmask = pushmask[0]
    return mask_idx, pullmask, pushmask
    
    
if __name__ == "__main__":  
    
    ## GLOABL VARS
    dff = True
    start = 0 # first frame in movie to detrend
    stop = 10000 # last frame to detrend
    window = 60 # window in seconds
    exposure = 10 # camera exposure in ms
    filepath = r'F:\150729-M187476\010101JCamF102_16_16_1_64x64.h5'
    output_file = r'F:\150729-M187476\010101JCamF102_16_16_1_64x64_detrend.h5'

    # load data
    open_tb = tb.open_file(filepath, 'r')
    mov = open_tb.root.data

    frames = range(start, stop)
    if stop > mov.shape[0]:
        frames = range(start, mov.shape[0])
    print str(len(frames)) + ' frames will be detrended'
    
    tdim = mov[frames,:,:].shape[0]
    ydim = mov.shape[1]
    xdim = mov.shape[2]
    
    # setup masking variables
    mask = get_mask(mov)
    masky = mask.shape[0]
    maskx = mask.shape[1]
    mask_idx, pullmask, pushmask = mask_to_index(mask)
    
    # open output file which will be a space x time transposition of the movie
    fd = tb.openFile(output_file, 'w')
    filters = tb.Filters(complevel=1, complib='blosc')
    mov_detrend = fd.createEArray(fd.root, 
                    'data', 
                    tb.Float32Atom(), 
                    expectedrows=pushmask.shape[0],
                    shape=(0, int(tdim)))
    
    # detrend the movie
    start_time = timeit.default_timer()
    detrend(mov, mov_detrend, mask_idx, pushmask, frames, exposure)
    detrend_time = timeit.default_timer() - start_time
    print 'detrending took ' + str(detrend_time) + ' seconds\n'