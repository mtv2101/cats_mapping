#!/shared/utils.x86_64/python-2.7/bin/python

"""
5/18/2017
mattv@alleninstitute.org
"""

# core
import os
import imp
import time
import timeit

# file i/o
import tables as tb

# math
import numpy as np
from skimage import transform
from scipy.ndimage.filters import gaussian_filter
from scipy.stats import linregress
from scipy.signal import butter, filtfilt
import cv2

# package dependencies
from cats_mapping.hemodynamics import crosstalk_2cam as crosstalk
from cats_mapping.hemodynamics.timealign_2cam import align_2cam as timealign
import corticalmapping.core.ImageAnalysis as ia

# Chris M's package dependencies (for affine registration)
os.chdir(r'\\allen\programs\braintv\workgroups\nc-ophys\test\automaticCorticalMappingAlignment')
alignment_path = r'\\allen\programs\braintv\workgroups\nc-ophys\test\automaticCorticalMappingAlignment\alignmentWrapper.py'
alignmentWrapper = imp.load_source('alignmentWrapper',alignment_path)
fileTools = imp.load_source('alignmentFileTools',alignment_path)


##############################################


class hemo_2cam(object):

	"""
	The hemo object has four video files:
			1. calib_cam1: 470nm fluorescence calibration movie (<2min) captured on reflectance camera (cam1)  
			2. calib_cam2: 470nm fluorescence calibration movie (<2min) captured on fluorescence camera (cam2)
			3. reflect: 575nm / 640nm reflectance experimental movie camptured on reflectance camera (cam1)
			4. fluor: 470 fluorescence experimental movie captured on fluorescence camera (cam2)

	Video in the hemo object is aligned in space and time:
		Time alignment uses vsync traces from a JPhys file to upsample 25Hz reflectance movies to the 100Hz fluorescence timebase
		Spatial alignment uses a rigid transformation found using SIFT

	Two short calibration movies as used to calculate channel cross-talk, and to calulate the rigid transform

	"""

	def __init__(self,
				align_cam1_path,
				align_cam2_path,
				calib_cam1_path,
				calib_cam2_path,
				reflect_path,
				fluor_path,
				jphys_path,
				output_filename = False):

		self.align_cam1_path = align_cam1_path
		self.align_cam2_path = align_cam2_path
		self.calib_cam1_path = calib_cam1_path
		self.calib_cam2_path = calib_cam2_path
		self.reflect_path = reflect_path
		self.fluor_path = fluor_path
		self.jphys_path = jphys_path
		self.output_filename = output_filename

	def load_h5(self, filepath):

		file_obj = tb.open_file(filepath, 'r')
		dat = file_obj.root.data
		print 'opening ' + str(filepath)
		print str(dat.shape) + ' ' + str(dat.dtype)

		return dat

	def butter_lowpass(self, sig, fs, corner, order=3):
	    nyq = 0.5 * fs
	    c = corner / nyq
	    b,a = butter(order, c, btype='lowpass', analog=False)
	    y1 = filtfilt(b, a, sig, axis=0)
	    return y1

	def make_mask(self, percentage=65.0):
		# mask fluorescence image using top "percentage" of pixels in raw brightness

		open_calib_cam2 = tb.open_file(self.calib_cam2_path, 'r')
		self.calib_cam2 = open_calib_cam2.root.data
		img_to_mask = self.calib_cam2[0,:,:]
		img_to_mask = img_to_mask.astype('uint16')

		img_dims = img_to_mask.shape
		bitdepth = 2**16
		img_hist, img_bins = np.histogram(img_to_mask, bitdepth/100, [0,bitdepth])
		background_proportion = (img_dims[0] * img_dims[1])/(100/percentage)
		cum = np.cumsum(img_hist)
		idx = cum[cum<background_proportion].shape[0]
		self.thresh = np.floor(img_bins[idx]).astype('uint16')

		mask = np.zeros(img_dims).astype('uint16')
		mask = mask.flatten()
		mask[img_to_mask.flatten()>self.thresh] = 1
		self.mask = mask.reshape(img_dims)

	def correct_crosstalk(self):
		### calculate average coefficient of cross-talk, within the mask, of fluorescence detected by cam2 bleeding through to cam1 

		temp = self.load_h5(self.calib_cam1_path)
		self.calib_cam2 = self.load_h5(self.calib_cam2_path)

		self.calib_cam1 = self.do_space_transform(temp)

		self.slope_map, self.intercept_map, self.r_value_map = self.do_linregress(self.calib_cam1, self.calib_cam2)

		ave_mask_slopes = np.nanmean(self.slope_map[self.mask==1].flatten())
		ave_mask_r = np.nanmean(self.r_value_map[self.mask==1].flatten())

		# remove fine spatial detail from regression coeffs.  This seems to improve the correction alot
		#self.mask_slopes_smooth = gaussian_filter(self.slope_map, 3.0)

		print 'Eliminating camera crosstalk: subtracting scaled fluorescence signal from reflectance channels'
		print 'average cross-talk coeff. in mask = ' + str(ave_mask_slopes)
		print 'average regression quality in mask = ' + str(ave_mask_r)

		#mov1_lp = self.butter_lowpass(self.mov1, 100.0, 3.0, order=5)

		self.mov2_ct = self.mov2 - ((self.mov1 * self.slope_map) + self.intercept_map)
		self.mov3_ct = self.mov3 - ((self.mov1 * self.slope_map) + self.intercept_map)

	def do_linregress(self, mov1, mov2):

		slope_map = np.zeros((mov1.shape[1],mov1.shape[2]))
		intercept_map = np.zeros((mov1.shape[1],mov1.shape[2]))
		r_value_map = np.zeros((mov1.shape[1],mov1.shape[2]))

		mov1 = np.copy(mov1).astype('float64')
		mov2 = np.copy(mov2).astype('float64')

		#mov1 -= np.mean(mov1, axis=0)
		#mov2 -= np.mean(mov2, axis=0)

		print mov1.shape
		print mov2.shape

		for ii in range(mov1.shape[1]):
		    for jj in range(mov1.shape[2]):
		        slope, intercept, r_value, p_value, std_err = linregress(mov2[:,ii,jj],mov1[:,ii,jj])
		        intercept_map[ii,jj] = intercept
		        slope_map[ii,jj] = slope
		        r_value_map[ii,jj] = r_value

		return slope_map, intercept_map, r_value_map

	def do_space_transform(self, target):
		# flip reflectance movie to match fluorescence movie, and then apply rigid transform for spatial alignment

		target_xform = np.zeros(target.shape)
		scale = (target.shape[1]*1.0)/(self.affineref.shape[1]*1.0)
		print 'transforming target movie shape from affine coordinates using scale = ' + str(scale)
		for n in range(target.shape[0]):
			target_xform[n,:,:] = np.fliplr(target[n,:,:])
			target_xform[n,:,:] = self.apply_affine(img=target_xform[n,:,:], M=self.affine_matrix, scale=scale)

		return target_xform

	def apply_affine(self,img,M,scale=1):
	    # from imaging_behavior.core.utilities.apply_affine
	    # written by DRO
		#apply the scale value to the translation components if it is not 1

		Mc = M.copy() #Note: we need to operate on a copy of M to avoid modifying the original in memory
		Mc[:,2] = Mc[:,2]*1.0*scale

		return cv2.warpAffine(img.astype(float),Mc,np.shape(img))

	def calc_affine(self):

		target = self.load_h5(self.align_cam1_path)[0,:,:]
		self.affinetarget = np.fliplr(target)
		self.affineref = self.load_h5(self.align_cam2_path)[0,:,:]

		self.affine_matrix, self.transform_image, self.abs_difference_image = alignmentWrapper.automatic_image_registration(self.affinetarget,self.affineref,cats=True,threshold_score=self.thresh)

		print 'affine matrix is:'
		print self.affine_matrix

	def mask_to_index(self):

		self.mask_idx = np.ndarray.nonzero(self.mask)
		self.pullmask = self.mask.reshape(self.mask.shape[0]*self.mask.shape[1]) #flatten image to vector
		self.pullmask = np.squeeze(np.array([self.pullmask==1])) #convert to boolean
		self.pushmask = np.ndarray.nonzero(self.pullmask) # indexes for within the mask 
		self.pushmask = self.pushmask[0]

	def mask_log_data(self):

		self.mask_to_index()

		mov1_ds = np.zeros([self.mov1.shape[0], len(self.pushmask)]).astype('float64')
		mov2_ds = np.zeros([self.mov2_ct.shape[0], len(self.pushmask)]).astype('float64')
		mov3_ds = np.zeros([self.mov3_ct.shape[0], len(self.pushmask)]).astype('float64')

		for n in range(len(self.pushmask)):
		    mov1_ds[:,n] = self.mov1[:, self.mask_idx[0][n], self.mask_idx[1][n]]
		    mov2_ds[:,n] = self.mov2_ct[:, self.mask_idx[0][n], self.mask_idx[1][n]]
		    mov3_ds[:,n] = self.mov3_ct[:, self.mask_idx[0][n], self.mask_idx[1][n]]

		self.mov1_log = np.log(mov1_ds/np.mean(mov1_ds, axis=0))
		self.mov2_log = np.log(mov2_ds/np.mean(mov2_ds, axis=0))
		self.mov3_log = np.log(mov3_ds/np.mean(mov3_ds, axis=0))

	def convert_units(self, mov):

		# subtract sensor offset, divide by sensor gain to convert counts to photo-electrons
		mov = (mov -100.0)/2.19
		return mov

	def package_data(self):

		fd = tb.open_file(self.output_filename, 'w')
		filters = tb.Filters(complevel=1, complib='blosc')

		# save JCam data (time, y, x)
		group = fd.create_group("/", 'JCam', 'channel')
		data_names = ['Fluorescence', 'Reflectance_575nm', 'Reflectance_640nm']

		for m,mov in enumerate([self.mov1, self.mov2_ct, self.mov3_ct]):

		    jcamdata = fd.create_earray(group, 
		                        data_names[m], 
		                        tb.UInt16Atom(), 
		                        expectedrows = int(mov.shape[0]),
		                        shape=(0, int(mov.shape[1]),int(mov.shape[2])))

		    for n in range(mov.shape[0]):     
		        frame = mov[n,:,:]
		        jcamdata.append(frame[None])

		# save masked JCam data (time, pixel)
		group = fd.create_group("/", 'JCam_log_masked', 'channel')
		data_names = ['Fluorescence', 'Reflectance_575nm', 'Reflectance_640nm']

		for m,mov in enumerate([self.mov1_log, self.mov2_log, self.mov3_log]):

		    jcamdata_log = fd.create_earray(group, 
		                        data_names[m], 
		                        tb.Float64Atom(), 
		                        expectedrows = int(mov.shape[0]),
		                        shape=(0, int(mov.shape[1])))

		    for n in range(mov.shape[0]):     
		        frame = mov[n,:]
		        jcamdata_log.append(frame[None])

		# save spatially aligned calibration JCam data (time, y, x)
		group = fd.create_group("/", 'spatially_aligned_calibration_data', 'channel')
		data_names = ['calib_cam1', 'calib_cam2']

		for m,mov in enumerate([self.calib_cam1, self.calib_cam2]):

		    jcamdata_calib = fd.create_earray(group, 
		                        data_names[m], 
		                        tb.Float64Atom(), 
		                        expectedrows = int(mov.shape[0]),
		                        shape=(0, int(mov.shape[1]),int(mov.shape[2])))

		    for n in range(mov.shape[0]):     
		        frame = mov[n,:,:]
		        jcamdata_calib.append(frame[None])

		# save JCam data before cross_talk correction(time, y, x)
#		group = fd.create_group("/", 'JCam_pre_crosstalk', 'channel')
#		data_names = ['Reflectance_575nm', 'Reflectance_640nm']
#
#		for m,mov in enumerate([self.mov2, self.mov3]):
#
#		    jcamdata_pre_ct = fd.create_earray(group, 
#		                        data_names[m], 
#		                        tb.UInt16Atom(), 
#		                        expectedrows = int(mov.shape[0]),
#		                        shape=(0, int(mov.shape[1]),int(mov.shape[2])))
#
#		    for n in range(mov.shape[0]):     
#		        frame = mov[n,:,:]
#		        jcamdata_pre_ct.append(frame[None])

		# save JPhys data
		jphysfile = np.fromfile(self.jphys_path, dtype=np.dtype('>f4'), count=-1)
		channelNum = 3
		channelLength = len(jphysfile) / channelNum
		jphysfile = jphysfile.reshape([channelLength, channelNum])
		basepath, jphys_name = os.path.split(self.jphys_path)
		jphysdata = fd.create_array(fd.root,                     
		                'jphys', 
		                jphysfile) 

		# save mask data
		maskdata = fd.create_array(fd.root, 
		            'mask', 
		            self.mask) 

		# save slope map
		slopedata = fd.create_array(fd.root, 
		            'slope_map', 
		            self.slope_map) 

		# save registration comparison
		slopedata = fd.create_array(fd.root, 
		            'abs_difference_image', 
		            self.abs_difference_image)

		# save registration SIFT vectors
		slopedata = fd.create_array(fd.root, 
		            'transform_image', 
		            self.transform_image)

		# save intercept_map
		slopedata = fd.create_array(fd.root, 
		            'intercept_map', 
		            self.intercept_map)

		affine_matrix = fd.create_array(fd.root,
					'affine_matrix', 
					self.affine_matrix)

		fd.close()

	def run_2cam(self):

		self.make_mask()

		# mov1 is fluorescence, mov2 and mov3 are two reflectance channels
		self.mov1, self.mov2, self.mov3 = timealign(self.fluor_path, self.reflect_path, self.jphys_path)

		# calculate rigid spatial registration from calibration movies
		self.calc_affine()

		# apply rigid spatial registration to reflectance movies 
		self.mov2 = self.do_space_transform(self.mov2)
		self.mov3 = self.do_space_transform(self.mov3)

		# remove the bleedthrough of cam2 fluorescence into cam1 reflectance
		self.correct_crosstalk()

		self.mov1 = self.convert_units(self.mov1)
		self.mov2_ct = self.convert_units(self.mov2_ct)
		self.mov3_ct = self.convert_units(self.mov3_ct)

		# reshape movie to (time, pixel), subtract sensor offset (100 count), convert to photo-electrons, and log transform the mean-normalized movie
		self.mask_log_data()

		if self.output_filename:
			self.package_data()


if __name__ == '__main__':

	print 'hemo_2cam.py'