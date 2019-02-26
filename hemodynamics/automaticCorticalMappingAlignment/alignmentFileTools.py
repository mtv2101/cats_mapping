# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 14:26:01 2015

@author: chrism
"""
#import cv2v3.cv2
#cv2 = cv2v3.cv2
##import cv2
##import cv2 as cv2
from import_shared_cv2 import CV2_DLL as cv2
import numpy as np
import scipy as sp
import cPickle as pickle

from toolbox.misc.slicer import BinarySlicer
import corticalmapping.core.ImageAnalysis as ia
from corticalmapping.HighLevel import translateMovieByVasculature as align_image
import corticalmapping.core.FileTools as ft

def arrayNor(A):
    """
    -normalize a np.array to the scale [0, 1]
    -modified function of Jun Zhuang
    """
    B=A.astype(np.float)
    B = (B-np.amin(B))/(np.amax(B)-np.amin(B))
    return B

def load_pkl(pkl_path):
    with open(pkl_path,"rb+") as f:
        pdata = pickle.load(f)
    return pdata
    
def load_mapping_img(pkl_path,img_key="vasculatureMap"):
    return load_pkl(pkl_path).get(img_key)

def load_experiment_img_jcamf(raw_exp_img_path,
                              exp_img_kwargs={"row":1024,"column":1024}):
    exp_img,_,_ = ft.importRawJCamF(raw_exp_img_path,**exp_img_kwargs)
    return exp_img[0]

def load_experiment_img(img_path,img_dims=(1024,1024),header=232,dtype="<u2"):
    return BinarySlicer(img_path,shape=(1,img_dims[0],img_dims[1]),
                        header=header,dtype=dtype)

def align_mapping_img(mapping_img,alignment_json,zoom=0.5):
    return ia.rigidTransform(align_image(mapping_img,alignment_json,1),
                             zoom=zoom)

def calc_ssid(img_a,img_b):
    """
    https://siddhantahuja.wordpress.com/tag/sum-of-squared-differences/
    """
    if img_a.dtype == np.uint8:
        img_a = img_a.astype(np.uint16)
    if img_b.dtype == np.uint8:
        img_b = img_b.astype(np.uint16)
    return sum(sum(cv2.absdiff(img_a,img_b)**2))

def calc_sid(img_a,img_b):
    """
    https://siddhantahuja.wordpress.com/tag/sum-of-squared-differences/
    """
    if img_a.dtype == np.uint8:
        img_a = img_a.astype(np.uint16)
    if img_b.dtype == np.uint8:
        img_b = img_b.astype(np.uint16)
    return sum(sum(cv2.absdiff(img_a,img_b)))

def box_enhanced_binary_erosion(img,erosion_kernel_size=(3,3),
                                box_kernel_size=(5,5),
                                box_depth=-1,
                                iterations=1):
    ret_img = np.array(img)
    
    erosion_kernel = np.ones(erosion_kernel_size,np.uint8) #binary erosion kernel
    box_kwargs = {
                "ddepth": box_depth,
                "ksize": box_kernel_size
    }
    for i in range(iterations):
        ret_img = cv2.erode(cv2.boxFilter(ret_img,**box_kwargs),erosion_kernel)
    return ret_img

def threshold_erode_img(img,thresh_args=(255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,31,3)):
    thresh = cv2.adaptiveThreshold(img,*thresh_args)
    erosion_kernel = np.ones((2,2),np.uint8) #binary erosion kernel
    return cv2.erode(thresh,erosion_kernel,iterations=1) 
    
def calc_thresholded_eroded_sid(img_a,img_b,n_erosions=3,replace_kernel=None,
                                thresh_args=[255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY_INV,55,3]):
    if replace_kernel:
        thresh_args[-2] = replace_kernel
    thresh_a = cv2.adaptiveThreshold(img_a,*thresh_args)
    thresh_b = cv2.adaptiveThreshold(img_b,*thresh_args)
    abs_diff = cv2.absdiff(thresh_a,thresh_b)
    erosion_kernel = np.ones((3,3),np.uint8)
    eroded_abs_diff = cv2.erode(abs_diff,erosion_kernel,iterations=n_erosions)
    
    return sum(sum(np.uint64(eroded_abs_diff)/255))


def convert_array_to_img_array_scipy(arr,dtype=np.uint8):
    return np.array(sp.misc.toimage(arr)).astype(dtype)

def convert_array_to_norm(arr,mval=255,dtype=np.uint8):
    norm_arr = arrayNor(arr)
    return (norm_arr*mval).astype(dtype)
    