# -*- coding: utf-8 -*-
"""
Created on Fri Jul 14 14:59:03 2017

@author: mattv
"""

from cats_mapping.hemodynamics.hemo_2cam import hemo_2cam


# 170524_M291236
hemo = hemo_2cam(r"Y:\CorticalMapping\IntrinsicImageData\170717-M296292\170717-M296292_cam1_crosstalkJCamF100.dcimg_4_4_1.h5",
                 r"Y:\CorticalMapping\IntrinsicImageData\170717-M296292\170717-M296292_cam2_crosstalkJCamF120.dcimg_4_4_1.h5",
                 r"Y:\CorticalMapping\IntrinsicImageData\170717-M296292\170717-M296292_cam1_crosstalkJCamF100.dcimg_16_16_1.h5",
                 r"Y:\CorticalMapping\IntrinsicImageData\170717-M296292\170717-M296292_cam2_crosstalkJCamF120.dcimg_16_16_1.h5",
                 r"Y:\CorticalMapping\IntrinsicImageData\170717-M296292\170717-M296292_cam1JCamF102.dcimg_16_16_1.h5",
                 r"Y:\CorticalMapping\IntrinsicImageData\170717-M296292\170717-M296292_cam2JCamF119.dcimg_16_16_1.h5",
                 r"Y:\CorticalMapping\IntrinsicImageData\170717-M296292\170717-M296292_cam1JPhys102",
                 r"Y:\CorticalMapping\IntrinsicImageData\170717-M296292\170717-M296292_alldat")

hemo.run_2cam()