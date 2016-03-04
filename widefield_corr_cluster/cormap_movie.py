#!/shared/utils.x86_64/python-2.7/bin/python

# written by Michael Buice

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os
import platform
if platform.system() == 'Linux': 
    sys.path.append(r'/data/nc-ophys/Matt/aibs/') # if windows local machine this should already be in base path    
import CorticalMapping.tifffile as tiff # need this to view df/f movie
import h5py

hdf5_format = True

if __name__=='__main__':

    if platform.system() == 'Windows':
        #basepath = r'\\aibsdata2\nc-ophys\Bethanny\141204'
        #basepath = r'C:\Users\supersub\Claire Dropbox\Dropbox\Matt\aibs work\141204'
        basepath = r'\\aibsdata2\nc-ophys\Matt\corrdata\M177929\20150501'        
        filename = '20150501JCamF100deci244_nonbehaving_cormap_tx10_sx2_lag_0_2015-05-22'
    elif platform.system() == 'Linux':
        basepath = r'/data/nc-ophys/Bethanny/141204/'
        filename = '141204JCamF106_cormap_tx4_sx2_2014-12-21.npy'

    def mask_to_index(mask):
        mask = mask.astype(np.intp) #64 bit integer
        mask_idx = np.ndarray.nonzero(mask)
        pullmask = mask.reshape(mask.shape[0]*mask.shape[1]) #flatten image to vector
        pullmask = np.squeeze(np.array([pullmask==1])) #convert to boolean
        pushmask = np.ndarray.nonzero(pullmask) # indexes for within the mask 
        pushmask = pushmask[0]
        return mask_idx, pullmask, pushmask
    
    def update_img(n):
        print n
        im.set_data(all_corr_mov[n,:,:])  #.T[::-1])             
        dot.set_data(mask_idx[1][n],mask_idx[0][n])
        return im
    
    #filename_short = '141204JCamF106deci244_cormap_tx10_sx4'
    #maskname = '141204JCamF106deci244_dffmovie_128_mask.tif' # mask size should match the final size of the correlation matrix
    maskname = 'mask256.tif'    
    filename_short = '20150501JCamF100deci244_nonbehaving_cormap_tx10_sx2' #string to use in creating filenames    
    maskpath = os.path.join(basepath,maskname)
    savefile = filename[:-4]+'.mp4'
    corr_path = os.path.join(basepath, 'correlations')
    movie_path = os.path.join(basepath, 'corr_movies')
    filepath = os.path.join(corr_path,filename)

    if hdf5_format:
        f = h5py.File(filepath, 'r')
        imported_data = f['all_corr']
        mov = np.copy(imported_data)    
    else:
        mov = np.load(filepath, 'r+')
  
    #mask = tiff.imread(maskpath) # mask must be 8bit thresholded as 0 & 255
    mask = tiff.imread(maskpath) # mask must be 8bit thresholded as 0 & 255
    mask = mask/255        
    masky = mask.shape[0]
    maskx = mask.shape[1]
    mask_idx, pullmask, pushmask = mask_to_index(mask)
    
    #mov[mov<0.75] = 0
    all_corr_mov = np.zeros([mov.shape[0], masky*maskx])
    all_corr_mov[:,pushmask] = mov
    all_corr_mov = all_corr_mov.reshape([mov.shape[0], masky, maskx])
    t,x,y = all_corr_mov.shape
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_aspect('equal')
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    fig.set_size_inches([10,10])
    im = ax.imshow(all_corr_mov[0].T[::-1],
                   interpolation='none', 
                   cmap='gnuplot2')
    im.set_clim(0,1)
    #coloraxis = fig.add_axes([0.92, 0.1, 0.03, 0.8]);
    #plt.colorbar(im, cax=coloraxis)
    dot, = plt.plot([],[], 'wo')
    numFrames = mov.shape[0]
    print 'rendering correlation movie frame #...'
    ani = animation.FuncAnimation(fig,
                                  update_img,
                                  frames=numFrames,
                                  blit=False)  #,interval=1000)
    writer = animation.writers['ffmpeg'](fps=30)
    savefile = os.path.join(movie_path, filename_short+'.mp4')
    ani.save(savefile,writer=writer,dpi=300) #max(x,y)/10)