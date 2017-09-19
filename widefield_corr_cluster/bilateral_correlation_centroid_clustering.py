#!/shared/utils.x86_64/python-2.7/bin/python

"""

Use DBSCAN or HDBSCAN algorithms to cluster features of the  
seed-pixel correlation matrix from movies of neural activity

Types of analyses:
1. Apply mulitple thresholds to correlation matrix
2. find centroids of thresholded correlation masks
3. cluster centroids using DBSCAN
4. collapse the correlation matrix using only seed-pixels that produce clustered centroids.

mattv@alleninstitute.org

"""

import numpy as np
import platform
import os
import sys
import time
import timeit
import cPickle
if platform.system() == 'Linux':
    import matplotlib
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
if platform.system() == 'Linux': 
    sys.path.append(r'/data/nc-ophys/Matt/') # if windows local machine this should already be in base path 
from pylab import cm
import aibs.CorticalMapping.tifffile as tiff
from sklearn.cluster import DBSCAN
import hdbscan
import h5py
from skimage.morphology import watershed
from skimage.feature import peak_local_max
from scipy import ndimage as ndi
import pdb


   

'''  '''
'''  '''   
'''  '''
   
   
   
   
class ccc(object): # correlate centroid clusters
    
    def __init__(self,
                 basepath, 
                 filename,
                 maskname,
                 method = 'HDBSCAN', # 'DBSCAN' or 'HDBSCAN', if 'HDBSCAN' eps is ignored
                 save_out = True,
                 corr_thresh = .8,
                 threshold_mult = 4.5, # multiple of mean corr to set as threshold 
                 eps=5,                                 # DBSCAN distance param
                 min_samples=200,                       # DBSCAN density param   
                 patch_size_threshold=200,    
                 hdf5_format=True,
                 do_canny=False, 
                 do_watershed=False
                ):
       
       self.basepath = basepath
       self.filename = filename
       self.method = method
       self.save_out = save_out
       self.threshold_mult = threshold_mult
       self.corr_thresh = corr_thresh
       self.eps = eps
       self.min_samples = min_samples
       self.patch_size_threshold = patch_size_threshold
       self.hdf5_format = hdf5_format
       self.do_canny = do_canny
       self.do_watershed = do_watershed
       self.maskname = maskname      
       
       mask, masky, maskx = ccc.get_masks(self)       
         
       self.mask = mask       
       self.masky = masky
       self.maskx = maskx
       self.blank_img = np.zeros([masky, maskx])
       self.blank_img_lin = np.zeros([masky*maskx])       
       self.pushmask = mask_to_index(mask)
       
       all_corr = ccc.get_data(self)
       self.all_corr = all_corr
       
       self.all_corr_img = ccc.all_corr_to_image(self)

       self.uber_centroid, self.area_masks, self.class_centroids, self.all_centroids, self.seed_pixel_masks = ccc.thresh_corr(self)
       
       self.areas = ccc.blend_images(self, self.area_masks)
       
       self.seed_pixel_maps = ccc.blend_images(self, self.seed_pixel_masks)
       
       ccc.pkl_out(self)
        
    @staticmethod
    def get_area_masks(self):
        return self.area_masks  
        
    @staticmethod
    def get_seedpixel_masks(self):
        return self.seed_pixel_masks
    
    @staticmethod   
    def get_corr_map_projections(self):
        return self.areas
        
    def pkl_out(self):
        if self.save_out:
            #save_path = os.path.join(self.basepath, 'ccc_output')
            #timestamp_str = time.strftime("%Y%m%d-%H%M")
            savename = '_ccc.pkl'
            pathout = self.basepath + savename
            with open(pathout, "wb") as output_file:
                cPickle.dump([self.area_masks, self.class_centroids, self.seed_pixel_masks], output_file)

    def get_data(self):
        ''' load correlation matrix ''' 
        #filepath = os.path.join(self.basepath,self.filename)
        if self.hdf5_format:
            f = h5py.File(self.filename, 'r')
            imported_data = f['all_corr']
            all_corr = np.copy(imported_data)
        else:            
            all_corr = np.load(self.filename, 'r') #load numpy array 
        return all_corr
        
    def get_masks(self):        
        ''' load masks ''' 
        #maskpath = os.path.join(self.basepath,self.maskname)
        #print maskpath
        mask = tiff.imread(self.maskname) # mask must be 8bit thresholded as 0 & 255
        mask = mask/255
        masky = mask.shape[0]
        maskx = mask.shape[1]
        return mask, masky, maskx 
    
    def thresh_corr(self):
        #print 'cluster and plot centroids\n'
        start_time = timeit.default_timer()

        lcy = np.zeros([self.pushmask.shape[0]])
        lcx = np.zeros([self.pushmask.shape[0]])          
        lcy, lcx, centroids_pixel, all_patches = ccc.get_centroid(self) # cy, cx are exact coordinates of centroid, centroids_pixel is the index in pushmask that gave that centroid                                   
        #pdb.set_trace()               
        all_centroids = np.column_stack((lcx, lcy, centroids_pixel))                                            
        
        uber_centroid, \
        class_centroids = ccc.cluster_using_DBSCAN(self,
                            np.squeeze(all_centroids))

        corr_map_areas = np.zeros([len(uber_centroid), self.masky, self.maskx])  
        seed_pixel_masks = np.zeros([len(uber_centroid), self.masky, self.maskx])
        for ii,cent in enumerate(uber_centroid): # for each cluster
            corr_map_areas[ii,:,:] = ccc.sum_thresholds(self, class_centroids[ii+1].values(), all_patches)
            seed_pixel_masks[ii,:,:] = ccc.seed_pixel_map(self, class_centroids[ii+1].values())
        
        corr_time = timeit.default_timer() - start_time
        print 'CPCC took ' + str(corr_time) + ' seconds\n'

        return uber_centroid, corr_map_areas, class_centroids, all_centroids, seed_pixel_masks
        
    def seed_pixel_map(self, class_centroids):
        ''' aggregate all seed-pixel coordinates for each cluster of centroids'''

        idxes = np.squeeze(np.array(class_centroids), axis=[0,1]) # extract np array from list
        if np.ndim(idxes) == 1:
            idxes = np.expand_dims(idxes, axis=0)
        seeds = idxes[:,2].astype(int) # convert the array indices (dim=3) to integers
        seed_pixels = np.copy(self.blank_img_lin)
        seed_pixels[self.pushmask[seeds]] = 1
        seed_pixels = seed_pixels.reshape(self.blank_img.shape)
        return seed_pixels        
 
    def get_centroid(self):
        ''' threshold correlation matrix and get centroid of the thresholded mask'''   
        patch_count = 0
        centroid_x = np.zeros((self.pushmask.shape[0]*4))
        centroid_y = np.zeros((self.pushmask.shape[0]*4))   
        centroids_pixel = np.zeros((self.pushmask.shape[0]*4))
        all_patches = np.zeros((self.pushmask.shape[0]*4,self.masky,self.maskx))
        adaptive_thresh = np.zeros(self.pushmask.shape[0])
                    
        for n in range(self.pushmask.shape[0]): 

            img = self.all_corr_img[n,:,:]
            #adaptive_thresh[n] = self.threshold_mult*np.nanmean(img)            
            ## hack override of adaptive trehsold ##
            adaptive_thresh[n] = self.corr_thresh
            ## hack override of adaptive trehsold ##
            #pdb.set_trace()
            corrmask = np.array(img>adaptive_thresh[n]) # do threshold     
            patch_array, number_patches = ndi.label(corrmask, np.ones((3,3)))
            
            if number_patches != 0:
                
                for jj in range(1,number_patches+1):
                    current_patch = np.zeros(self.mask.shape)
                    current_patch[patch_array == jj] = 1 # make binary mask just from the patch
                    
                    if np.sum(current_patch.flatten()) <= self.patch_size_threshold:  # size filter 
                        
                        if self.do_watershed:
                            distance = ndi.distance_transform_edt(current_patch)
                            local_maxi = peak_local_max(distance, indices=False, min_distance=100, labels=current_patch)
                            markers = ndi.label(local_maxi, np.ones((3,3)))[0]
                            num_watershed = np.max(markers)
                            labels = watershed(-distance, markers, mask=current_patch)
                            
                            # get centroids for each sub-patch
                            for ii in range(1,num_watershed+1):
                                sub_patch = np.zeros(self.mask.shape)
                                sub_patch[labels == ii] = 1        
                                
                                if np.sum(sub_patch.flatten()) >= self.min_samples / 2: # size filter, bias towards halving
                                    all_patches[patch_count,:,:] = sub_patch * corrmask                                
                                    # calculate the centroid of each patch, or sub patch (if watersheded)        
                                    idx_x, idx_y = np.nonzero(sub_patch) # get all indices in patch
                                    centroid_x[patch_count] = np.nanmean(idx_x) # x centroid
                                    centroid_y[patch_count] = np.nanmean(idx_y) # y centroid
                                    centroids_pixel[patch_count] = n
                                    patch_count += 1
                        
                        else: #if no watershed
                            all_patches[patch_count,:,:] = current_patch * corrmask                            
                            # calculate the centroid of each patch, or sub patch (if watersheded)        
                            idx_x, idx_y = np.nonzero(current_patch) # get all indices in patch
                            centroid_x[patch_count] = np.nanmean(idx_x) # x centroid
                            centroid_y[patch_count] = np.nanmean(idx_y) # y centroid
                            centroids_pixel[patch_count] = n
                            patch_count += 1
           
        self.adaptive_thresh = adaptive_thresh
        
        # cutoff excess pre-allocation
        all_patches = all_patches[:patch_count,:,:]
        centroid_x = centroid_x[:patch_count]
        centroid_y = centroid_y[:patch_count]
        centroids_pixel = centroids_pixel[:patch_count]
        print 'patch_count = ' + str(patch_count)
        if patch_count == 0:
            print 'no patches found that meet the size requirement, please increase min_samples'
        
        return centroid_y, centroid_x, centroids_pixel, all_patches  
     
    def cluster_using_DBSCAN(self, all_centroids):
        '''use DBSCAN algorithm to cluster centroids'''
        
        if self.method == 'DBSCAN':
            db_cores, centroid_labels = ccc.dbscan_centroids(self, all_centroids)   
        elif self.method == 'HDBSCAN':
            centroid_labels = ccc.hdbscan_centroids(self, all_centroids) 
        #pdb.set_trace()     
        labels = [s for i,s in enumerate(set(centroid_labels)) if s != -1] # get label categories 
        self.labels = labels
        
        ##now group seedpixels, core-pixels and centroids according to label
        class_centroids = []
        class_centroids.append({'all_centroids' : [all_centroids]}) # create list of dicts to collect all class_centroids per class      
        #pdb.set_trace() 
        for n, cat in enumerate(labels):      
                           
            #core_idx = [ii for i,ii in enumerate(db_cores) if centroid_labels[ii] == cat] # index of centroid list per cluster 
            core_idx = [i for i,ii in enumerate(centroid_labels) if ii == cat]        
            label_centroids = all_centroids[core_idx] 
            core_idx = np.expand_dims(np.array(core_idx),1) 
            label_centroids = np.hstack((label_centroids, core_idx)) # make fourth column of label_centroids its index in the centroid list    
            class_centroids.append({n : [label_centroids]})
            if n == 0:
                uber_centroid = [(np.ceil(np.nanmean(label_centroids[:,0])), np.ceil(np.nanmean(label_centroids[:,1])))]
            else: 
                uber_centroid += [(np.ceil(np.nanmean(label_centroids[:,0])), np.ceil(np.nanmean(label_centroids[:,1])))]

        return uber_centroid, class_centroids 
        
    def mask_to_img(self, m):
        img = np.zeros([self.masky*self.maskx])
        img[:] = np.nan
        img[self.pushmask] = m
        img = img.reshape([self.masky,self.maskx])
        return img
        
    def all_corr_to_image(self):
        n_samp = self.all_corr.shape[0]
        dat = np.zeros([n_samp, self.masky*self.maskx])
        dat[:] = np.nan
        dat[:,self.pushmask] = self.all_corr
        dat = dat.reshape([n_samp, self.masky, self.maskx])
        return dat
        
    def dbscan_centroids(self, dat):
        '''DBSCAN clustering'''
        '''http://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html''' 
        
        db_model = DBSCAN(self.eps, self.min_samples).fit(dat[:,0:2])
        db_cores = db_model.core_sample_indices_
        #pdb.set_trace()
        centroid_labels = db_model.labels_
        #centroid_cores = dat[db_cores,:]
        return db_cores, centroid_labels
        
    def hdbscan_centroids(self, dat):
        ''' heirarchical DBSCAN '''
        '''https://github.com/lmcinnes/hdbscan'''
        #pdb.set_trace()
        db_model = hdbscan.HDBSCAN(self.min_samples, gen_min_span_tree=True)
        centroid_labels = db_model.fit_predict(dat[:,0:2])
        #pdb.set_trace() 
        #centroid_labels = db_model.labels_
        return centroid_labels
        
    def sum_thresholds(self, sp_mask, all_patches): 
        ''' all_patches are the corr masks, sp_mask is [xidx, yidx, pushmask idx, patch idx] '''

        all_idxes = np.squeeze(np.array(sp_mask), axis=[0,1]) # extract np array from list
        if np.ndim(all_idxes) == 1:
            all_idxes = np.expand_dims(all_idxes, axis=0)
        patch_idxes = all_idxes[:,3].astype(int) # convert the array indices (dim=3) to integers
        areas = all_patches[patch_idxes,:,:] # pull out just the patches in the cluster       
        sum_areas = np.squeeze(np.nansum(areas, axis=0))
        norm = np.nanmax(sum_areas)#-(np.nanstd(proj_areas)/3)
        sum_areas = sum_areas/norm 
        return sum_areas
        
    def blend_images(self, data): #data must have iterable 0th dimension
        ''' RGBA blending of cluster projections '''
        blended = np.zeros([self.masky,self.maskx,4])
        if np.ndim(data) == 2:
            composite = np.zeros([self.masky,self.maskx,4])
            color = cm.hsv(.1)
            alpha = np.squeeze(data)
            alpha[np.isnan(alpha)] = .001 #zap nans
            alpha[alpha==0] = .001
            blended = rgba_blend(composite, color, alpha)
        elif np.ndim(data) == 3: 
            composite = np.zeros([self.masky,self.maskx,4,data.shape[0]])
            composite[:] = .001
            color_scale = np.arange(.1,1,.9/data.shape[0])
            for ii in range(data.shape[0]): # for each cluster build a unique color channel  
                toget = color_scale[ii]
                color = cm.hsv(toget)
                alpha = np.copy(data[ii,:,:])
                alpha[np.isnan(alpha)] = .001 #zap nans
                alpha[alpha==0] = .001 # np.average requires non-zero weights
                composite[:,:,:,ii] = rgba_blend(composite, color, alpha)
            for ii in range(3): # for each color make weighted average
                color = composite[:,:,ii,:]
                alpha = composite[:,:,3,:]
                blended[:,:,ii] = np.average(color, axis=2, weights=alpha)
            blended[:,:,3] = np.nanmax(composite[:,:,3,:], axis=2)
        return blended
        
    def plot_class_centroids(self, dat, color, ax, hexbin=False):
        
        if hexbin==True:
            plt.xlim(0,self.maskx)
            plt.ylim(0,self.masky)  
            ax.hexbin(dat[:,1], self.masky-dat[:,0], gridsize=32, cmap=color, vmax=self.min_samples)
        else:
            ax.imshow(self.mask, cmap='gray_r', vmax=10, interpolation='nearest')
            plt.xlim(0,self.maskx)
            plt.ylim(0,self.masky)
            ax.set_ylim(ax.get_ylim()[::-1])
            if np.ndim(dat) == 1:
                ax.scatter(dat[1], dat[0], s=10, color=color, marker='.', alpha=0.5, edgecolor='none')
            elif np.ndim(dat) == 2:
                ax.scatter(dat[:,1], dat[:,0], s=10, color=color, marker='.', alpha=0.5, edgecolor='none')
        
    def plot_all(self, dat1, dat2, dat3, dat4):

        font = {'family' : 'sans-serif',
        'fontname' : 'Arial',
        'color'  : 'black',
        'weight' : 'normal',
        'size'   : 10,
        }
        
        text_ycords = self.masky *.05
        text_xcords = self.maskx *.04
        fig = plt.figure(figsize = (16,4));      
        fig.subplots_adjust(left=.05, right=.95, bottom=.05, top=.9, hspace=.1, wspace=.05)   
        color_scale = np.arange(.1,1,.9/len(self.labels))
        
        ax1 = fig.add_subplot(1, 4, 1, xticks=[], yticks=[]);
        ccc.plot_class_centroids(self, dat1, cm.gist_heat_r, ax1, hexbin=True)
        ax1.set_title('all centroids', fontdict=font, fontsize=14)
        ax1.text(text_xcords, text_ycords*19, 'threshold: ' + str(self.corr_thresh), fontdict=font)
        ax1.text(text_xcords, text_ycords*18, 'density: ' + str(self.eps), fontdict=font)
        ax1.text(text_xcords, text_ycords*17, 'min. size: ' + str(self.min_samples), fontdict=font)
        
        ax2 = fig.add_subplot(1, 4, 2, xticks=[], yticks=[]);
        for ii in range(len(dat2)): # dat2 is one longer tha dat1 because it contains unused centroids
            dat2_vals = np.squeeze(np.asarray(dat2[ii].values()))
            if ii==0:   # first entry is unused centroids              
                ccc.plot_class_centroids(self, dat2_vals, cm.gray_r(.3), ax2)
            else:
                ccc.plot_class_centroids(self, dat2_vals, cm.hsv(color_scale[ii-1]), ax2)
        ax2.set_title('clustered centroids', fontdict=font, fontsize=14)
        ax2.text(text_xcords, text_ycords, 'threshold: ' + str(self.corr_thresh), fontdict=font)
        ax2.text(text_xcords, text_ycords*2, 'density: ' + str(self.eps), fontdict=font)
        ax2.text(text_xcords, text_ycords*3, 'min.size: ' + str(self.min_samples), fontdict=font)

        ax3 = fig.add_subplot(1, 4, 3, xticks=[], yticks=[]);
        ax3.imshow(dat3, interpolation='nearest')
        ax3.set_title('projected correlation maps', fontdict=font, fontsize=14)
        ax3.text(text_xcords, text_ycords, 'threshold: ' + str(self.corr_thresh), fontdict=font)
        ax3.text(text_xcords, text_ycords*2, 'density: ' + str(self.eps), fontdict=font)
        ax3.text(text_xcords, text_ycords*3, 'min.size: ' + str(self.min_samples), fontdict=font)
        
        ax4 = fig.add_subplot(1, 4, 4, xticks=[], yticks=[]);
        ax4.imshow(dat4, interpolation='nearest')
        ax4.set_title('seed-pixel masks', fontdict=font, fontsize=14)
        ax4.text(text_xcords, text_ycords, 'threshold: ' + str(self.corr_thresh), fontdict=font)
        ax4.text(text_xcords, text_ycords*2, 'density: ' + str(self.eps), fontdict=font)
        ax4.text(text_xcords, text_ycords*3, 'min.size: ' + str(self.min_samples), fontdict=font)
        
        #timestamp_str = time.strftime("%Y%m%d-%H%M")
        if self.save_out:
            savename = self.basepath + '_ccc.png'
            fig.savefig(savename, dpi=300, format='png')


def mask_to_index(mask):
    ''' find indices of image that are in mask (pushmask) '''        
    mask = mask.astype(np.intp) #64 bit integer
    pullmask = mask.reshape(mask.shape[0]*mask.shape[1]) #flatten image to vector
    pullmask = np.squeeze(np.array([pullmask==1])) #convert to boolean
    pushmask = np.nonzero(pullmask) # indexes for within the mask 
    pushmask = pushmask[0]
    return pushmask
    
def rgba_blend(img, c2, alpha):
    out = np.zeros([img.shape[0], img.shape[1], 4])
    out[:] = 0.001    
    for ii in range(img.shape[0]):
        for jj in range(img.shape[1]):
            out[ii,jj,3] = alpha[ii,jj]
            out[ii,jj,0]  = c2[0]
            out[ii,jj,1]  = c2[1]
            out[ii,jj,2]  = c2[2]
    return out
  
        
        
if __name__ == "__main__":
    

    basepath = r'\\aibsdata2\nc-ophys\CorticalMapping\IntrinsicImageData\160412-M210101-Retinotopy'
    filename = r'\\aibsdata2\nc-ophys\CorticalMapping\IntrinsicImageData\160412-M210101-Retinotopy\corr\160412JCamF101.dcimg_2_2_10_cormap__tx10_sx4_nframes_all_dff_True2016-04-14'
    maskname = r'\\aibsdata2\nc-ophys\CorticalMapping\IntrinsicImageData\160412-M210101-Retinotopy\mask_128.tif'
    
    cc = ccc(basepath, filename, maskname, method='HDBSCAN', corr_thresh = .8, eps=4, min_samples=100)
    
    area_masks = cc.get_area_masks(cc)
    
    areas = cc.get_corr_map_projections(cc)
    
    seed_pixel_masks = cc.get_seedpixel_masks(cc)
    
    cc.plot_all(cc.all_centroids, cc.class_centroids, cc.areas, cc.seed_pixel_maps)