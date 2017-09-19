# -*- coding: utf-8 -*-
"""
Created on Thu May 25 12:21:57 2017

@author: mattv
"""

from cats_mapping.hemodynamics.hemo_2cam import hemo_2cam


# 170524_M291236
hemo = hemo_2cam(r"Y:\CorticalMapping\IntrinsicImageData\170524_M295771\170524_M295771_calib_cam1JCamF102.dcimg_8_8_1.h5",
        r"Y:\CorticalMapping\IntrinsicImageData\170524_M295771\170524_M295771_calib_cam2JCamF103.dcimg_8_8_1.h5",
        r"Y:\CorticalMapping\IntrinsicImageData\170524_M295771\170524_M295771_cam1JCamF101.dcimg_8_8_1.h5",
        r"Y:\CorticalMapping\IntrinsicImageData\170524_M295771\170524_M295771_cam2JCamF104.dcimg_8_8_1.h5",
        r"Y:\CorticalMapping\IntrinsicImageData\170524_M295771\170524_M295771_cam1JPhys101",
        r"Y:\CorticalMapping\IntrinsicImageData\170524_M295771\170524_M295771_alldat")

hemo.run_2cam()