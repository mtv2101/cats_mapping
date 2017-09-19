# -*- coding: utf-8 -*-
"""
Created on Thu May 25 12:21:57 2017

@author: mattv
"""

from cats_mapping.hemodynamics.hemo_2cam import hemo_2cam


# 170524_M291236
hemo = hemo_2cam(r"Y:\CorticalMapping\IntrinsicImageData\170605-M291236\170605-M291236_cam1_calibJCamF100.dcimg_4_4_1.h5",
                 r"Y:\CorticalMapping\IntrinsicImageData\170605-M291236\M291236_cam2_calibJCamF102.dcimg_4_4_1.h5",
                 r"Y:\CorticalMapping\IntrinsicImageData\170605-M291236\170605-M291236_cam1_calibJCamF100.dcimg_16_16_1.h5",
                 r"Y:\CorticalMapping\IntrinsicImageData\170605-M291236\M291236_cam2_calibJCamF102.dcimg_16_16_1.h5",
                 r"Y:\CorticalMapping\IntrinsicImageData\170605-M291236\M291236_cam1JCamF100.dcimg_16_16_1.h5",
                 r"Y:\CorticalMapping\IntrinsicImageData\170605-M291236\M291236_cam2JCamF103.dcimg_16_16_1.h5",
                 r"Y:\CorticalMapping\IntrinsicImageData\170605-M291236\M291236_cam1JPhys100",
                 r"Y:\CorticalMapping\IntrinsicImageData\170605-M291236\170605-M291236_alldat_test")

hemo.run_2cam()