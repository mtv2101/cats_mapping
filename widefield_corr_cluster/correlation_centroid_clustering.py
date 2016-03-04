#!/shared/utils.x86_64/python-2.7/bin/python

"""

Use the DBSCAN algorithm to cluster features of the  
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
if platform.system() == 'Linux':
    import matplotlib
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
if platform.system() == 'Linux': 
    sys.path.append(r'/data/nc-ophys/Matt/aibs/') # if windows local machine this should already be in base path 
from pylab import cm
from matplotlib.colors import LinearSegmentedColormap
import aibs.CorticalMapping.tifffile as tiff
from sklearn.cluster import DBSCAN
import h5py
from skimage import feature
import pdb
import matplotlib.gridspec as gridspec
from skimage.morphology import watershed
from skimage.feature import peak_local_max
from scipy import ndimage as ndi


   

'''  '''
'''  '''   
   
   
   
   
class ccc(object): # correlate centroid clusters
    
    def __init__(self,
                 basepath, 
                 filename,
                 maskname,
                 masknameleft,
                 masknameright,
                 corr_thresh=.8, # correlation threshold 
                 eps=3,                                 # DBSCAN distance param
                 min_samples=200,                       # DBSCAN density param       
                 hdf5_format=True,
                 do_canny=False, 
                 do_watershed=False
                ):
       
       self.basepath = basepath
       self.filename = filename
       self.corr_thresh = corr_thresh
       self.eps = eps
       self.min_samples = min_samples
       self.hdf5_format = hdf5_format
       self.do_canny = do_canny
       self.do_watershed = do_watershed
       self.maskname = maskname
       self.masknameleft = masknameleft
       self.masknameright = masknameright       
       
       mask, masky, maskx, maskleft, maskright = ccc.get_masks(self)       
         
       self.mask = mask       
       self.masky = masky
       self.maskx = maskx
       self.maskleft = maskleft
       self.maskright = maskright
       self.blank_img = np.zeros([masky, maskx])
       self.blank_img_lin = np.zeros([masky*maskx])
       
       self.pushmask = mask_to_index(mask)
       l_pushmask = mask_to_index(maskleft)
       r_pushmask = mask_to_index(maskright)
       self.l_pushmask = l_pushmask
       self.r_pushmask = r_pushmask
       
       all_corr = ccc.get_data(self)
       self.all_corr = all_corr
       
       self.all_corr_img = ccc.all_corr_to_image(self)

       self.area_masks, self.class_centroids_l, self.label_map = ccc.thresh_corr(self)
       
       self.areas = ccc.blend_images(self, self.area_masks)
       
       self.binmasks = ccc.areas_to_binarymasks(self)

        
    @staticmethod
    def get_area_masks(self):
        return self.area_masks        
    
    @staticmethod   
    def get_corr_map_projections(self):
        return self.areas
        
#    def sum_corr_maps(self, area_masks):
#        num_masks = area_masks.shape[0]
#        outlines = np.zeros([num_masks, self.masky, self.maskx])
#        #areas = np.zeros([num_masks, self.masky, self.maskx])
#
#        for jj in range(num_masks): # for each mask         
#            outlines[jj,:,:], areas[jj,:,:] = ccc.sum_thresholds(self, self.area_masks[jj,:,:])
#        
#        composite_edges = ccc.blend_images(self, outlines)            
#        composite_areas = ccc.blend_images(self, self.area_masks)            
#
#        return composite_edges, composite_areas

    def get_data(self):
        ''' load correlation matrix '''
        
        filepath = os.path.join(self.basepath,self.filename)
        
        if self.hdf5_format:
            f = h5py.File(filepath, 'r')
            imported_data = f['all_corr']
            all_corr = np.copy(imported_data)
        else:            
            all_corr = np.load(filepath, 'r') #load numpy array 

        return all_corr
        
    def get_masks(self):        
        ''' load masks '''
        
        maskpath = os.path.join(self.basepath,self.maskname)
        
        mask = tiff.imread(maskpath) # mask must be 8bit thresholded as 0 & 255
        mask = mask/255
        masky = mask.shape[0]
        maskx = mask.shape[1]
        
        maskpathleft = os.path.join(self.basepath,self.masknameleft)
        maskpathright = os.path.join(self.basepath,self.masknameright)
        maskleft = tiff.imread(maskpathleft)
        maskleft = maskleft/255
        maskright = tiff.imread(maskpathright)  
        maskright = maskright/255
        
        return mask, masky, maskx, maskleft, maskright
    
    
    def thresh_corr(self):
        print 'cluster and plot centroids'
        #pdb.set_trace()
        all_centroids_l, all_centroids_r, submask_l, submask_r = ccc.thresh_corr_centroids(self)                                                      
        
        submask_l, \
        submask_r, \
        uber_centroid_l, \
        uber_centroid_r, \
        all_label_map_l, \
        all_label_map_r, \
        class_centroids_l = ccc.cluster_using_DBSCAN(self,
                            np.squeeze(all_centroids_l), 
                            np.squeeze(all_centroids_r))
        
        areas = np.zeros([len(uber_centroid_l), self.masky, self.maskx])   
        #pdb.set_trace()
        for ii, idx in enumerate(uber_centroid_l):
            #pushmask_idx = self.masky*idx[0]+idx[1] #get linear index of uber centroid
            lin_idx = all_label_map_l[ii,:] == 1
            #lin_idx = [nn for nn in range(self.pushmask.shape[0]) if self.pushmask[nn] == pushmask_idx] # find index of pushmask corresponding to uber_centroid
            
            #pullmask = self.mask.reshape(self.mask.shape[0]*self.mask.shape[1]) 
            #pullmask = np.squeeze(np.array([pullmask==1])) #get boolean image of mask            
            uber_masks = self.all_corr_img[lin_idx[self.pushmask]]    # uber masks are all the masks from clustered seed-pixels
            uber_masks[uber_masks<self.corr_thresh] = np.nan
            uber_masks[uber_masks>self.corr_thresh] = 1
            areas[ii,:,:] = ccc.sum_thresholds(self, uber_masks)
        
        # finally reshape lables back to image space
        if np.ndim(all_label_map_l) == 2:
            all_label_map_l = all_label_map_l.reshape([all_label_map_l.shape[0],self.masky, self.maskx])
        elif np.ndim(all_label_map_l) == 1:
            all_label_map_l = all_label_map_l.reshape([self.masky, self.maskx]) 
        
        return areas, class_centroids_l, all_label_map_l

    def thresh_corr_centroids(self):
        print 'get left centroids from each seed-pixel'

        submask_l = [n for n in range(self.pushmask.shape[0]) if self.pushmask[n] in self.l_pushmask] # indices of pushmask that are also in l_pushmask
        submask_r = [n for n in range(self.pushmask.shape[0]) if self.pushmask[n] in self.r_pushmask]
        lcy = np.zeros([self.l_pushmask.shape[0]])
        lcx = np.zeros([self.l_pushmask.shape[0]])
        rcy = np.zeros([self.r_pushmask.shape[0]])
        rcx = np.zeros([self.r_pushmask.shape[0]])
        all_centroids_l = np.zeros([self.l_pushmask.shape[0], 2])
        all_centroids_r = np.zeros([self.r_pushmask.shape[0], 2])
        
        all_centroid_image = self.blank_img
        
        for n in range(self.l_pushmask.shape[0]):                  
            lcy[n], lcx[n], idx = ccc.get_centroid(self, self.all_corr_img[submask_l[n],:,:]*self.maskleft) # cy, cx are exact coordinates of centroid, idx is nearest index (rounded towards bottom right pixel)
            all_centroid_image[idx] = all_centroid_image[idx] + 1 
                                                 
        for n in range(self.r_pushmask.shape[0]):   
            rcy[n], rcx[n], idx = ccc.get_centroid(self, self.all_corr_img[submask_r[n],:,:]*self.maskright) 
            all_centroid_image[idx] = all_centroid_image[idx] + 1
                     
        all_centroids_l = np.column_stack((lcx, lcy))
        all_centroids_r = np.column_stack((rcx, rcy))    
    
        return all_centroids_l, all_centroids_r, submask_l, submask_r
 
    def get_centroid(self, img):
        ''' threshold correlation matrix and get centroid of the thresholded mask'''        
        
        idx_x = []
        idx_y = []
        corrmask = np.array(img>self.corr_thresh) # do threshold
        #pdb.set_trace()
        
        if self.do_watershed:
            distance = ndi.distance_transform_edt(corrmask)
            local_maxi = peak_local_max(distance, indices=False, footprint=np.ones((10, 10)),labels=corrmask)
            markers = ndi.label(local_maxi)[0]
            labels = watershed(-distance, markers, mask=corrmask)
        
        idx_x, idx_y = np.nonzero(img>self.corr_thresh) # get all indices of non-thresholded pixels
        corrmask = np.ones(self.mask.shape, dtype='int32')*corrmask #turn boolean mask into integers
        centroid_x = np.nanmean(idx_x)
        centroid_x_idx = np.ceil(centroid_x).astype('int16')
        centroid_y = np.nanmean(idx_y)
        centroid_y_idx = np.ceil(centroid_y).astype('int16')
        centroid_idx_out = (centroid_x_idx, centroid_y_idx)
        return centroid_y, centroid_x, centroid_idx_out     
     
    def cluster_using_DBSCAN(self, all_centroids_l, all_centroids_r):
        print 'use DBSCAN algorithm to cluster centroids'
    
        submask_l = [n for n in range(self.pushmask.shape[0]) if self.pushmask[n] in self.l_pushmask] # indices of pushmask that are also in l_pushmask
        submask_r = [n for n in range(self.pushmask.shape[0]) if self.pushmask[n] in self.r_pushmask]

        l_centroid_cores, l_db_cores, l_centroid_labels = ccc.dbscan_centroids(self, all_centroids_l)   
        r_centroid_cores, r_db_cores, r_centroid_labels = ccc.dbscan_centroids(self, all_centroids_r)
                
        #l_seeds = [submask_l[cent] for i,cent in enumerate(l_db_cores)]
        #r_seeds = [submask_r[cent] for i,cent in enumerate(r_db_cores)]
        #used_seed_pixels_cores = l_seeds + r_seeds
        
        l_centroid_labels = l_centroid_labels.tolist()
        r_centroid_labels = r_centroid_labels.tolist()
    
        labels_l = [s for i,s in enumerate(set(l_centroid_labels)) if s != -1] # get label categories
        labels_r = [s for i,s in enumerate(set(r_centroid_labels)) if s != -1]    
        self.labels_l = labels_l 
        self.labels_r = labels_r
        
        ##now group seedpixels, core-pixels and centroids according to label

        class_centroids_l = []
        class_centroids_l.append({'all_centroids' : [all_centroids_l]}) # create list of dicts to collect all class_centroids per class
        
        for n, cat in enumerate(labels_l):  
            #print 'getting seed-pixels that produce clustered centroids'                      
            core_idx = [ii for i,ii in enumerate(l_db_cores) if l_centroid_labels[ii] == cat] # index of all core seed-pixels
            #allcore_idx = [ii for i,ii in enumerate(l_centroid_labels) if ii == cat] 
            label_centroids = all_centroids_l[core_idx]
            label_map = np.zeros([self.masky*self.maskx])
            label_map[:] = np.nan
            label_map[self.l_pushmask[core_idx]] = 1      
            class_centroids_l.append({n : [label_centroids]})
            if n == 0:
                uber_centroid_l = [(np.ceil(np.nanmean(label_centroids[:,0])), np.ceil(np.nanmean(label_centroids[:,1])))]
                all_label_map_l = np.copy(label_map)
            else: 
                uber_centroid_l += [(np.ceil(np.nanmean(label_centroids[:,0])), np.ceil(np.nanmean(label_centroids[:,1])))]
                all_label_map_l = np.vstack((all_label_map_l, label_map))
                
        for n, cat in enumerate(labels_r): 
            core_idx = [ii for i,ii in enumerate(r_db_cores) if r_centroid_labels[ii] == cat] # index of all core seed-pixels
            label_centroids = all_centroids_r[core_idx]
            label_map = np.zeros([self.masky*self.maskx])
            label_map[:] = np.nan
            label_map[self.r_pushmask[core_idx]] = 1
            if n == 0:
                uber_centroid_r = [(np.ceil(np.nanmean(label_centroids[:,0])), np.ceil(np.nanmean(label_centroids[:,1])))]
                all_label_map_r = np.copy(label_map)
            else: 
                uber_centroid_r += [(np.ceil(np.nanmean(label_centroids[:,0])), np.ceil(np.nanmean(label_centroids[:,1])))]   
                all_label_map_r = np.vstack((all_label_map_r, label_map))             

        return submask_l, submask_r, uber_centroid_l, uber_centroid_r, all_label_map_l, all_label_map_r, class_centroids_l
        
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
        db_model = DBSCAN(self.eps, self.min_samples).fit(dat)
        db_cores = db_model.core_sample_indices_
        centroid_labels = db_model.labels_
        centroid_cores = dat[db_cores,:]
        return centroid_cores, db_cores, centroid_labels
        
    def sum_thresholds(self, sp_mask): # sp_mask are the corr masks from al clustered seed-pixels
        #sp_mask = sp_mask.reshape([self.masky*self.maskx])
        #mask_idx = [i for i,push in enumerate(self.pushmask) if ~np.isnan(sp_mask[push])] # linear indices of the clustered seed-pixels within the mask
        #all_corr_mov = self.all_corr_img[mask_idx,:,:].reshape([len(mask_idx), self.masky*self.maskx]) # apply mask to seed-pixel dimension and reshape correlation dimenisons    
        
#        mov_edges = [ccc.get_edges(self, sp_mask[n,:], self.corr_thresh) for n in range(all_corr_mov.shape[0])]        
#        proj_edges = np.nansum(mov_edges, axis=0) 
#        norm = np.nanmax(proj_edges)#-(np.nanstd(proj_edges))
#        proj_edges = proj_edges/norm
#        proj_edges[proj_edges>1] = 1 # for alpha mapping cannot have values above 1
#        proj_edges = proj_edges.reshape([self.masky, self.maskx])
        
        proj_areas = ccc.sum_areas(self, sp_mask, self.corr_thresh)
        norm = np.nanmax(proj_areas)#-(np.nanstd(proj_areas)/3)
        proj_areas = proj_areas/norm
        proj_areas[proj_areas>1] = 1 # for alpha mapping cannot have values above 1
        #proj_areas = proj_areas.reshape([self.masky, self.maskx])
        
        return proj_areas
        
    def sum_areas(self, mov, thresh):
        
        mov[mov <= thresh] = 0
        sum_areas = np.squeeze(np.nansum(mov, axis=0))
        return sum_areas
        
    def get_edges(self, mov, thresh): 
        #pdb.set_trace()
        
        if self.do_canny:
            thresh_img = self.blank_img_lin
            thresh_img[mov>thresh] = 1
            thresh_img = thresh_img.reshape([self.masky, self.maskx])
            edges = feature.canny(thresh_img, 3, np.max(thresh_img)/20, np.max(thresh_img)/10) 
            out = self.blank_img
            out[edges] = 1 # convert boolean to binary
        else:
            #pdb.set_trace()
            thresh_img = np.copy(self.blank_img_lin)
            thresh_idx = [i for i,val in enumerate(mov) if val > thresh-.01 and val < thresh+.01]
            thresh_img[thresh_idx] = 1
            out = thresh_img
            
        return out
        
    def blend_images(self, data): #data must have iterable 0th dimension
        #pdb.set_trace()
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
                alpha = data[ii,:,:]
                alpha[np.isnan(alpha)] = .001 #zap nans
                alpha[alpha==0] = .001
                composite[:,:,:,ii] = rgba_blend(composite, color, alpha)
            for ii in range(3): # for each color make weighted average
                color = composite[:,:,ii,:]
                alpha = composite[:,:,3,:]
                blended[:,:,ii] = np.average(color, axis=2, weights=alpha)
            blended[:,:,3] = np.nanmax(composite[:,:,3,:], axis=2)
        return blended
        
    def plot_class_centroids(self, dat, color, ax):
        ax.imshow(self.mask, cmap='gray_r', vmax=10, interpolation='nearest')
        plt.xlim(0,self.maskx)
        plt.ylim(0,self.masky)
        ax.set_ylim(ax.get_ylim()[::-1])
        if np.ndim(dat) == 1:
            ax.scatter(dat[1], dat[0], s=10, color=color, marker='.', alpha=0.5, edgecolor='none')
        elif np.ndim(dat) == 2:
            ax.scatter(dat[:,1], dat[:,0], s=10, color=color, marker='.', alpha=0.5, edgecolor='none')
        
    def plot_all(self, dat1, dat2, dat3, dat4):
        #pdb.set_trace()
        fig = plt.figure(figsize = (12,4));      
        fig.subplots_adjust(left=.05, right=.95, bottom=.05, top=.9, hspace=.05, wspace=.05)   
        color_scale = np.arange(.1,1,.9/len(self.labels_l))
        
        ax1 = fig.add_subplot(1, 4, 1, xticks=[], yticks=[]);
        for ii in range(len(dat1)): # dat1 is one longer tha dat2 because it contains unused centroids
            dat1_vals = np.squeeze(np.asarray(dat1[ii].values()))
            if ii==0:   # first entry is unused centroids              
                ccc.plot_class_centroids(self, dat1_vals, cm.gray_r(.3), ax1)
            else:
                ccc.plot_class_centroids(self, dat1_vals, cm.hsv(color_scale[ii-1]), ax1)
        ax1.set_title('clustered centroids')
        
        ax2 = fig.add_subplot(1, 4, 2, xticks=[], yticks=[]);

        composite = ccc.blend_images(self, dat2) 
        ax2.imshow(self.mask, cmap='gray_r', vmax=10, interpolation='nearest')
        ax2.imshow(composite, interpolation='nearest')
        ax2.set_title('clustered seed-pixels')
        
        ax3 = fig.add_subplot(1, 4, 3, xticks=[], yticks=[]);
        ax3.imshow(dat3, interpolation='nearest')
        ax3.set_title('projected correlation maps')
        
        ax4 = fig.add_subplot(1, 4, 4, xticks=[], yticks=[]);
        ax4.imshow(dat4, interpolation='nearest')
        ax4.set_title('correlation masks')
            
    def areas_to_binarymasks(self):        
        binmask = np.copy(self.area_masks)
        binmask[binmask>self.corr_thresh] = 1
        binmask[binmask<self.corr_thresh] = np.nan
        binmask_blend = ccc.blend_images(self, binmask)
        return binmask_blend


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
    
    basepath = r'F:\150804-M187474\correlations'
    
    filename = '010101JCamF102_2_2_1_cormap_tx10_sx2_lag_0_2015-09-25'
    maskname = 'mask_256.tif'
    masknameleft = 'mask_256_left.tif'
    masknameright = 'mask_256_right.tif'
    
#    filename = '20150225JCamF102deci244_cormap_tx500_sx8_lag_0_2015-05-02'
#    maskname = '102mask_64.tif'
#    masknameleft = '102mask_64_left.tif'
#    masknameright = '102mask_64_right.tif'
    
    cc = ccc(basepath, filename, maskname, masknameleft, masknameright)
    
    area_masks = cc.get_area_masks(cc)
    
    areas = cc.get_corr_map_projections(cc)
    
    cc.plot_all(cc.class_centroids_l, cc.label_map, cc.areas, cc.binmasks)
