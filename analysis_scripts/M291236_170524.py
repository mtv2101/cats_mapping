# -*- coding: utf-8 -*-
"""
Created on Thu May 25 12:21:57 2017

@author: mattv
"""

from cats_mapping.hemodynamics.hemo_2cam import hemo_2cam


# 170524_M291236
hemo = hemo_2cam(r"Y:\CorticalMapping\IntrinsicImageData\170531-M291236\M291236_170531_cam1_crosstalkJCamF100.dcimg_4_4_1.h5",
                 r"Y:\CorticalMapping\IntrinsicImageData\170531-M291236\M291236_170531_cam2_crosstalkJCamF106.dcimg_4_4_1.h5",
                 r"Y:\CorticalMapping\IntrinsicImageData\170531-M291236\M291236_170531_cam1_crosstalkJCamF100.dcimg_16_16_1.h5",
                 r"Y:\CorticalMapping\IntrinsicImageData\170531-M291236\M291236_170531_cam2_crosstalkJCamF106.dcimg_16_16_1.h5",
                 r"Y:\CorticalMapping\IntrinsicImageData\170531-M291236\M291236_170531_cam1JCamF102.dcimg_16_16_1.h5",
                 r"Y:\CorticalMapping\IntrinsicImageData\170531-M291236\M291236_170531_cam2_JCamF107.dcimg_16_16_1.h5",
                 r"Y:\CorticalMapping\IntrinsicImageData\170531-M291236\M291236_170531_cam1JPhys102",
                 r"Y:\CorticalMapping\IntrinsicImageData\170531-M291236\170531_M291236_alldat_test")

hemo.run_2cam()