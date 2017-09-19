#!/shared/utils.x86_64/python-2.7/bin/python

"""

Jcorr:

1. Load a movie and a binary mask of values 0,255
2. Downsample movie in x/y
3. Use per-pixel gaussian filter as f0 to calculate df/f
4. Bandpass filter timeseries
5. Do temporal correlation within mask pixels using random subset of movie frames

November, 2015

mattv@alleninstitute.org

"""


import os, sys
import platform
import time
import datetime
import gc
import numpy as np
import h5py
import scipy.signal as sig
import timeit
import tables as tb
from skimage.measure import block_reduce
if platform.system() == 'Linux':
    import matplotlib
    matplotlib.use('Agg')
if platform.system() == 'Linux': 
    sys.path.append(r'/data/nc-ophys/Matt/') # if windows local machine this should already be in base path    
    sys.path.append(r'/data/nc-ophys/Matt/imagingbehavior') # if windows local machine this should already be in base path    
    import aibs.CorticalMapping.tifffile as tiff # need this to view df/f movieexit
    import imaging_behavior
    from imaging_behavior.core.slicer import BinarySlicer
else:
    import aibs.CorticalMapping.tifffile as tiff
    
import pdb




'''  '''
'''  '''   
'''  '''
   
   
   
   
class Jcorr(object): 

    ''' 
    filepath: [str] path to movie

    savepath: [str] path to save correlation matrix

    rsamp_factor: [int] build correlation matrix using random samples this factor of total samples.  This random selection happens after all filtering and normalization of the movie

    dsfactor: [int] spatial downsample factor

    frameNum: [int] or [array] can be integer indexing array, single integer "n" to pull first "n" frames, or "-1" to get all frames 

    window: [int] gaussian smoothing filter, this is kernal length in seconds.  Gaussian is ~kernal/8 FWHM.  Only used if "moving_dff" = True

    lag: [int] lag in frames used for cross-correlation.  Default=0

    exposure: [int], in ms

    img_source: 'npy': numpy array dims [time, y, x]
                'h5': pytables hdf5 file saved in root 
                'old_camera': old mapping Flash binary JCam file
                'mapping_tif': tiff stack
                'orca_flash': Flash 4.0 binary JCam file

    do_corr: if True build correlation matrix

    importmask: if False will auto generate a mask using intensity values in the first frame

    dff: if True will calculate df/f, if False will calculate df.  The f0 is caluclated using the method specified in "moving_dff"

    SAVE_DFF: if True will re-save the whole movie after normalization (df or dff)

    do_bandpass: if True will bandpass filter the movie at multiple bands between 0.1 Hz and 30Hz and make a correlation matrix from each band

    moving_dff: if True will de-trend the movie using gaussian convolution length "window"

    special_string: a string added to filenames upon saving 

    '''
    
    def __init__(self,
                 filepath,
                 savepath,
                 rsamp_factor,
                 dsfactor,
                 absmaskpath = '',
                 frameNum = -1, 
                 window = 60,
                 lag = 0,
                 exposure = 10, 
                 img_source = 'h5',
                 do_corr = True,
                 importmask = False, 
                 dff = True,
                 SAVE_DFF = False,
                 do_bandpass = False,
                 do_bandcorr = True,
                 moving_dff = False, 
                 special_string = ''
                 ):
                 
        self.filepath = filepath       
        self.savepath = savepath
        self.basepath, self.filename = os.path.split(filepath)
        self.filename_short = self.filename[:-4] # cut off '.npy'
        
        self.rsamp_factor = rsamp_factor
        self.dsfactor = dsfactor
        self.frameNum = frameNum
        self.window = window
        self.lag = lag
        self.exposure = exposure
        self.img_source = img_source
        self.do_corr = do_corr
        self.do_bandcorr = do_bandcorr
        self.importmask = importmask
        self.dff = dff        
        self.SAVE_DFF = SAVE_DFF
        self.do_bandpass = do_bandpass
        self.moving_dff = moving_dff
        
        self.timestamp_str = str(datetime.datetime.now()) # this may be added to save strings later
        self.special_string = special_string

        self.absmaskpath = absmaskpath
        self.maskname = 'mask128.tif'
           
        self.mask, self.maskpath = Jcorr.load_mask(self)        
        self.masky = self.mask.shape[0]
        self.maskx = self.mask.shape[1]
        
        #pullmask: boolean array defining mask
        #pushmask: indexes that project masked data back to full image dimensions (linearlized)
        self.mask_idx, self.pullmask, self.pushmask = Jcorr.mask_to_index(self, self.mask)
        self.len_iter = self.pushmask.shape[0]          
        
        #print 'rsamp_factor = ' + str(rsamp_factor)
        #print 'dsfactor = ' + str(dsfactor)

        if np.size(frameNum) > 1: # if index used to get frames
            frame_str = str(np.size(frameNum))
            print 'frameNum = ' + frame_str
        elif frameNum == -1:
            frame_str = 'all'
            print 'frameNum = ' + frame_str
        elif np.size(frameNum) == 1:
            frame_str = str(frameNum)  
            print 'frameNum = ' + frame_str    
        else:
            frame_str = ''   
        
        #print 'window = ' + str(window)
        #print 'lag = ' + str(lag)
        #print 'SAVE_DFF = ' + str(SAVE_DFF)
        #print 'do_corr = ' + str(do_corr) 
        #print 'img_source = ' + str(img_source)
        
        # analysis paramater string to add to saved filenames
        self.filenamenote = '_tx' + str(rsamp_factor) + '_sx' + str(dsfactor) + '_nframes_' + frame_str + '_dff_' + str(dff) + self.special_string # tx is time downsample, sx is space downsample
        print self.filenamenote
           
           
           
    @staticmethod
    def return_corr(self):        

        if self.do_bandpass:        
            self.all_corr = Jcorr.bandpass_movie(self)        
        else:        
            self.all_corr = Jcorr.simple_correlate(self)   
            
        return self.all_corr
        
    def ImportMappingTif(self, path): #used to import tiff stacks, most often with vasculature maps
        imageFile = tiff.imread(path)
        exposureTime = 10
        return imageFile, exposureTime
        
    def importRawJCam(self, path,
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
        
    def importMov(self, path, frameNum): 
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
        
    def mask_to_index(self, mask):
        mask = mask.astype('uint16') #64 bit integer
        mask_idx = np.ndarray.nonzero(mask)
        pullmask = mask.reshape(mask.shape[0]*mask.shape[1]) #flatten image to vector
        pullmask = np.squeeze(np.array([pullmask==1])) #convert to boolean
        pushmask = np.ndarray.nonzero(pullmask) # indexes for within the mask 
        pushmask = pushmask[0]
        return mask_idx, pullmask, pushmask
        
    def do_subsample(self, mov):
        print 'subsampling movie...\n'
        mov_tiny = np.zeros([mov.shape[0],mov.shape[1]/self.dsfactor, mov.shape[2]/self.dsfactor], dtype=('float'))
        for i in xrange(mov_tiny.shape[-2]):
            for j in xrange(mov_tiny.shape[-1]):
                mov_tiny[:,i,j] = mov[:, i*self.dsfactor:(i+1)*self.dsfactor,j*self.dsfactor:(j+1)*self.dsfactor].mean(-1).mean(-1)
        return mov_tiny
    
    def do_moving_dff(self, mov):    
        # define size of rolling average 
        expose = np.float(self.exposure/1000.) #convert exposure from ms to s
        win = np.float(self.window) # window in seconds
        win = win/expose
        win = np.ceil(win) 
        if win > mov.shape[0]/2:
            print 'please choose a window smaller than half the length of time you are analyzing'
        # pre-allocate matrices
        kernal = sig.gaussian(win, win/8)
        padsize = mov.shape[0] + (win*2) - 1
        mov_pad = np.zeros([padsize], dtype=('float32'))
        #mov_ave = np.zeros([padsize+win-1], dtype=('float'))
        mov_dff = np.zeros([mov.shape[0], mov.shape[1], mov.shape[2]], dtype=('float32')) 
        #delta_map = np.zeros([mov.shape[1], mov.shape[2]])
        # baseline subtract by gaussian convolution along time (dim=0)   
        yidx = np.array(self.mask_idx[0])
        xidx = np.array(self.mask_idx[1])
        print 'per-pixel baseline subtraction using gaussian convolution ...\n'
        #pdb.set_trace()
        print str(self.len_iter) + ' pixels'
        for n in range(self.len_iter):       
            print 'baseline subtraction pixel ' + str(n)
            # put data in padded frame
            mov_pad = Jcorr.pad_vector(self, mov[:,yidx[n],xidx[n]], win)
            # moving average by convolution
            mov_ave = sig.fftconvolve(mov_pad, kernal/kernal.sum(), mode='valid')
            # cut off pad
            mov_ave = mov_ave[win/2:(-win/2)-1]
            mov_ave = mov_ave.astype('float32')
            #if n == 0:
                #mov_ave_accum = np.zeros(mov_ave.shape[0])
            #else:
                #mov_ave_accum = ((mov_ave_accum*(n-1))+mov_ave)/n 
            #delta_map[yidx[n],xidx[n]] = np.mean(mov_ave[:win])-np.mean(mov_ave[-win:])
            # and now use moving average as f0 for df/f
            if self.dff:
                mov_dff[:,yidx[n],xidx[n]] = (mov[:,yidx[n],xidx[n]] - mov_ave)/mov_ave
            else:
                mov_dff[:,yidx[n],xidx[n]] = (mov[:,yidx[n],xidx[n]] - mov_ave)
                
        return mov_dff#, mov_ave_accum, delta_map

    def do_moving_dff_transpose(self, mov):    
        # define size of rolling average 
        expose = np.float(self.exposure/1000.) #convert exposure from ms to s
        win = np.float(self.window) # window in seconds
        win = win/expose
        win = np.ceil(win) 
        if win > mov.shape[1]/2:
            print 'please choose a window smaller than half the length of time you are analyzing'
        # pre-allocate matrices
        kernal = sig.gaussian(win, win/8)
        padsize = mov.shape[1] + (win*2) - 1
        mov_pad = np.zeros([padsize], dtype=('float32'))
        # pre-allocate the filtered matrix
        mov_dff = np.zeros([mov.shape[0], mov.shape[1]], dtype=('float32'))
        # baseline subtract by gaussian convolution along time (dim=0) 
        print 'per-pixel baseline subtraction using gaussian convolution ...\n'
        print str(self.len_iter) + ' pixels'
        for n in range(self.len_iter):    
            print 'baseline subtraction pixel ' + str(n)
            # put data in padded frame
            mov_seg = np.squeeze(mov[n,:])
            mov_pad = Jcorr.pad_vector(self, mov_seg, win)
            # moving average by convolution
            mov_ave = sig.fftconvolve(mov_pad, kernal/kernal.sum(), mode='valid')
            # cut off pad
            mov_ave = mov_ave[win/2:(-win/2)-1]
            mov_ave = mov_ave.astype('float32')
            # and now use moving average as f0 for df/f
            if self.dff:
                mov_dff[n, :] = (mov[n, :] - mov_ave)/mov_ave
            else:
                mov_dff[n, :] = (mov[n, :] - mov_ave) 

        return mov_dff
    
    def pad_vector(self, dat, win):
        tlen = dat.shape[0]      
        pad_start = dat[0:win]+(dat[0]-dat[win])
        pad_end = dat[tlen-win:]+(dat[-1]-dat[tlen-win])
        dat_pad = np.append(np.append(pad_start, dat), pad_end)
        return dat_pad
    
    def add_dims(self, seed_pixel_ts, sliced_mov_dff):
        for d in range(sliced_mov_dff.ndim-1):
            seed_pixel_ts = np.expand_dims(seed_pixel_ts,d+1)
        return seed_pixel_ts   
            
    def smart_imshow_cscale(self, plot_probs):
        plot_probs_lin = np.reshape(plot_probs, [self.masky*self.maskx])
        img_median = np.nanmedian(plot_probs_lin)
        img_mad = np.nanmedian(np.absolute(plot_probs_lin - np.nanmedian(plot_probs_lin))) #mean absolute deviation
        upper_clim = img_median+(img_mad*3)
        lower_clim = img_median-(img_mad*3)
        return lower_clim, upper_clim
        
    def rand_tsamp(self, mov):
        r_subsamp = np.ceil(mov.shape[1]/self.rsamp_factor)
        r_subsamp = r_subsamp.astype('uint16')
        ridx = np.random.randint(0, mov.shape[1], r_subsamp)
        mov = mov[:,ridx]
        return mov
        
    def get_seed_pixel_ts(self, lag, sliced_mov_dff, seed_idx): # 
        if lag < 0: 
            lag_abs = np.abs(lag)
            seed_pixel_ts = np.pad(sliced_mov_dff[:-lag_abs, seed_idx], [lag_abs, 0], mode='median')
        elif lag > 0:
            seed_pixel_ts = np.pad(sliced_mov_dff[lag:, seed_idx], [0, lag], mode='median')
        elif lag == 0:
            seed_pixel_ts = sliced_mov_dff[seed_idx,:]
        return seed_pixel_ts
    
    def FIR_bandpass(self, lowcut, highcut, fs, filt_length):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        a = sig.firwin(filt_length, low, window='blackmanharris')
        b = - sig.firwin(filt_length, high, window = 'blackmanharris'); 
        b[filt_length/2] = b[filt_length/2] + 1
        d = - (a+b); # combine lowpass and highpass into bandpass
        d[filt_length/2] = d[filt_length/2] + 1
        return d
        
    def FIR_lowpass(self, cut, fs, filt_length):
        nyq = 0.5 * fs
        corner = cut / nyq
        h = sig.firwin(filt_length, corner, window='hamming')
        return h
        
    def do_correlation(self, mov_dff):        
        
        start_time = timeit.default_timer()
        
        all_corr = np.zeros([self.len_iter, self.len_iter], dtype='float')
        corr_constant = np.sqrt(np.sum(mov_dff*mov_dff,axis=1), dtype='float')   
        cor = np.zeros([self.len_iter])
        for n in range(mov_dff.shape[0]):
            all_corr[n,:] = Jcorr.corr_map(self, mov_dff, corr_constant, n, cor) # do the correlation
        all_corr = np.array(all_corr) #convert from list of arrays to 2-d numpy array
       
        corr_time = timeit.default_timer() - start_time
        print 'correlation took ' + str(corr_time) + ' seconds\n'
        
        return all_corr

    def corr_map(self, sliced_mov_dff, corr_constant, seed_idx, cor): 
        #pdb.set_trace()
        seed_pixel_ts = sliced_mov_dff[seed_idx,:]
        # do cross-correlation of seed-pixel 
        cor = (seed_pixel_ts*sliced_mov_dff).sum(axis=1)
        cor = cor/corr_constant
        cor = np.squeeze(cor/np.sqrt(np.dot(seed_pixel_ts,seed_pixel_ts)))

        #idx = np.float(seed_idx)
        #progress = (idx/self.len_iter)*100.
        #print '\rCorrelating: %' + str(progress)
       
        return cor
        
    def save_corr(self, all_corr, path):
        output = h5py.File(path, 'w')
        output.create_dataset('all_corr', data=all_corr)
        output.close()
    
    def generate_mask(self, img, percentage):
        
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
    
    def load_mov(self, frameNum):
        ''' Load JCam movie in memory '''    
        if self.img_source == 'old_camera':
            mov, exposure = self.importRawJCam(self.filepath)
        elif self.img_source == 'mapping_tif':
            mov, exposure = self.ImportMappingTif(self.filepath)
        elif self.img_source == 'orca_flash':
            mov, exposure = self.importMov(self.filepath, frameNum) # set frameNum = -1 to load whole file
        elif self.img_source == 'npy':
            mov = np.load(self.filepath, 'r') 
        elif self.img_source == 'h5':
            open_tb = tb.open_file(self.filepath, 'r')
            mov = open_tb.root.data
        elif self.img_source == 'slicer':
            if frameNum != -1:
                mov = BinarySlicer(self.filepath)[:frameNum,:,:]
            else:
                mov = BinarySlicer(self.filepath)
                
        return mov
     
    def load_mask(self):
       ''' mask Jcam to isolate the brain from background '''
       if self.absmaskpath:
           print 'loading mask from ' + str(self.absmaskpath)
           mask = tiff.imread(self.absmaskpath) # mask must be 8bit thresholded as 0 & 255
           mask = mask/255
           mask_savepath = self.absmaskpath
       else: # create mask automatically from top %50 pixels in histogram
           if self.img_source == 'npy': #cut off file extension
               mask_savepath = os.path.join(self.savepath, self.filename[:-4]+ '_' + self.special_string + self.maskname)
           else:
               mask_savepath = os.path.join(self.savepath, self.filename+ '_' + self.special_string + self.maskname)
               open_tb = tb.open_file(self.mask_moviepath, 'r')
               mask_mov = open_tb.root.data
           if len(np.shape(mask_mov)) == 2:
               frame = mask_mov[:,:]
           else:
               frame = mask_mov[0,:,:]
           if self.dsfactor != 1:
               frame =  block_reduce(frame, block_size=(self.dsfactor,self.dsfactor), func=np.mean)
           del mask_mov    
           mask = Jcorr.generate_mask(self, frame, 50)           
           #mask_tosave = mask.astype('uint8')
           #mask_tosave = mask_tosave*255 # for historical reasons, the mask is either 0 and 255
           #tiff.imsave(mask_savepath, mask_tosave)
       return mask, mask_savepath

    def load_subsample_movie(self):    
        
        if self.dsfactor == 1:
            mov = Jcorr.load_mov(self, self.frameNum)
            if np.size(self.frameNum) == 1 and self.frameNum != -1: # i.e. frameNum = 1000, get first 1000 frames
                mov = mov[:,:self.frameNum]
            elif np.size(self.frameNum) > 1 or self.frameNum == -1: # if -1, or int array, do nothing
                print 'loading whole time series'
                pass
        else:
            mov =  Jcorr.load_mov(self, self.frameNum)
            if np.size(self.frameNum) == 1 and self.frameNum != -1: # i.e. frameNum = 1000, get first 1000 frames
                mov = mov[:,:self.frameNum]
                mov = Jcorr.do_subsample(self, mov)                 # then downsample in space from the small array
            elif self.frameNum == -1 or np.size(self.frameNum) > 1: # if -1, or int array, do nothing
                mov = Jcorr.do_subsample(self, mov)

        return mov
        
    def bandpass_movie(self):
        
        # load and spatially downsample movie
        mov = Jcorr.load_subsample_movie(self)
        print 'movie shape = ' + str(mov.shape)

        if np.size(self.frameNum) > 1:
            mov_idx = np.arange(self.frameNum[0], self.frameNum[-1])
            idx_gaps = self.frameNum - self.frameNum[0] # if index has gaps, ignore them for now, and cut them out after filtering
            idx_gaps = idx_gaps[:-1]
            print 'extracting ' + str(mov_idx.shape[0]) + ' frames for filtering'
            start_time = timeit.default_timer()
            mov = mov[:, mov_idx] # converts to numpy array
            load_time = timeit.default_timer() - start_time
            print 'loading into numpy took ' + str(load_time) + ' seconds\n'

        # first highpass using gaussian filter before doing bandpass
        #print 'filtering with sliding gaussian ' + str(nn)
        #start_time = timeit.default_timer()
        #mov_dff = Jcorr.do_moving_dff(self, mov)
        #mov_dff = Jcorr.do_moving_dff_transpose(self, mov)
        #filt_time = timeit.default_timer() - start_time
        #print 'filter took ' + str(filt_time) + ' seconds\n'
        #del mov

        # build corr matrix
        if self.do_corr:
            # extract samples to be used for building correlation matrix
            if np.size(self.frameNum) > 1: # now excise the gaps from the frame index
                mov_dff_trunc = mov[:, idx_gaps]
                print 'extracting ' + str(mov_dff_trunc.shape[1]) + ' frames for correlation'
                all_corr = Jcorr.do_correlation(self, mov_dff_trunc)
            else:
                all_corr = Jcorr.do_correlation(self, mov)
    
            corrsavepath = os.path.join(self.savepath,self.filename_short+'_cormap_'+self.filenamenote)
            Jcorr.save_corr(self, all_corr, corrsavepath)
        else:
            all_corr = []
    
        fs = 100.
        bands = [[.1, .5, 2048], # low pass, highpass in Hz, filter length in samples  
                [1, 5, 512]]  
        
        for nn, band in enumerate(bands):
            print 'filtering band ' + str(nn)
            start_time = timeit.default_timer()
            mov_dff_bp = np.zeros(mov.shape, dtype='float32') # suplicate mov in memory for bandpass array
            coeffs = Jcorr.FIR_bandpass(self, band[0], band[1], fs, band[2])
            for n in range(self.len_iter):
                #print n
                 out = sig.filtfilt(coeffs, 1, mov[n,:]) # use default pad values
                 out = np.float32(out)
                 mov_dff_bp[n,:] = out
            corr_time = timeit.default_timer() - start_time
            print 'filter took ' + str(corr_time) + ' seconds\n'

            if np.size(self.frameNum) > 1: # now excise the gaps from the frame index
                mov_dff_bp = mov_dff_bp[:, idx_gaps]

            bandpassnote = self.filenamenote + 'low' + str(band[0]) + '_high' + str(band[1]) + '_'        
            if self.SAVE_DFF:
                #dffsavepath = os.path.join(self.savepath,self.filename_short+'_movie_'+self.filenamenote+bandpassnote+self.timestamp_str[:10])
                dffsavepath = os.path.join(self.savepath, self.filename_short+bandpassnote + 'dff')
                np.save(dffsavepath, mov_dff_bp)

            if self.rsamp_factor != 1:    
                mov_dff_bp = Jcorr.rand_tsamp(self, mov_dff_bp)

            if self.do_bandcorr:
                # build corr matrix
                all_corr = Jcorr.do_correlation(self, mov_dff_bp)
                
                corrsavepath = os.path.join(self.savepath,self.filename_short+bandpassnote+'cormap')
                Jcorr.save_corr(self, all_corr, corrsavepath)

            del mov_dff_bp
            
        return all_corr
        
    def simple_correlate(self):
        
        mov = Jcorr.load_subsample_movie(self)

        if self.SAVE_DFF:
            dffsavepath = os.path.join(self.savepath, self.filename_short+ '_dff_' +self.filenamenote)
            np.save(dffsavepath, mov)

        #pdb.set_trace()
        if np.size(self.frameNum) > 1: # now excise the gaps from the frame index
            mov = mov[:,self.frameNum]
            print 'extracting ' + str(mov.shape[1]) + ' frames for correlation'

#        if self.rsamp_factor != 1:    
#            mov = Jcorr.rand_tsamp(self, mov)

        if self.do_corr:
            all_corr = Jcorr.do_correlation(self, mov)
            
            corrsavepath = os.path.join(self.savepath,self.filename_short+'_cormap_'+self.filenamenote)
            #corrsavepath = self.savepath + self.filenamenote + '_cormap_' + self.timestamp_str[:10]
            Jcorr.save_corr(self, all_corr, corrsavepath)
        else:
            all_corr = []
        
        return all_corr
          
          
          
if __name__ == "__main__":  
 
    #filepath = r'\\aibsdata2\nc-ophys\CorticalMapping\IntrinsicImageData\150731-M187476\010101JCamF102_16_16_1.npy'
    filepath = r'F:\150729-M187476\010101JCamF102_16_16_1_64x64_detrend_transpose.h5'    
    savepath = r'F:\150729-M187476'
    #savepath = r'\\aibsdata2\nc-ophys\CorticalMapping\IntrinsicImageData\150731-M187476'

    #Jcorr(filepath, savepath, rsamp_factor, dsfactor)    
    corr = Jcorr(filepath,savepath, 1, 1, moving_dff = False, do_corr=True)
       
    all_corr = corr.return_corr(corr)                                                                                              