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
import scipy.fftpack
import scipy.signal as sig
import scipy.ndimage as ndimage
from skimage import transform
from skimage.measure import block_reduce

def load_data(path1, path2):

	dat1 = tb.open_file(path1, 'r')
	mov1 = dat1.root.data
	print 'loading ' + str(path1)
	print 'size ' + str(mov1.shape)

	dat2 = tb.open_file(path2, 'r')
	mov2 = dat2.root.data
	print 'loading ' + str(path2)
	print 'size ' + str(mov2.shape)

	return mov1, mov2

def format_data(mov1, mov2, affine_coords=None):

	mov1_mean = np.mean(mov1, axis=0)
	mov2_mean = np.mean(mov2, axis=0)

	mov1 -= mov1_mean
	mov2 -= mov2_mean

	return mov1, mov2

def linregress(mov1_ds, mov2_ds):

	slope_map = np.zeros((mov1_ds.shape[1],mov1_ds.shape[2]))
	r_value_map = np.zeros((mov1_ds.shape[1],mov1_ds.shape[2]))

	for ii in range(mov1_ds.shape[1]):
	    for jj in range(mov1_ds.shape[2]):

	        slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(mov2_ds[:,ii,jj],mov1_ds[:,ii,jj])
	        slope_map[ii,jj] = slope
	        r_value_map[ii,jj] = r_value

	return slope_map, r_value_map

def calc_crosstalk(path1, path2, affine_coords, savepath=False):

	mov1, mov2 = load_data(path1, path2)

	mov1_ds, mov2_ds = format_data(mov1, mov2, affine_coords)

	slope_map, r_value_map = linregress(mov1_ds, mov2_ds)

	if savepath:
		np.save(savepath, slope_map)

	return slope_map, r_value_map