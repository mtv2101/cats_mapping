# -*- coding: utf-8 -*-
"""
Created on Thu May 25 12:21:57 2017

@author: mattv
"""

from cats_mapping.hemodynamics.hemo_2cam import hemo_2cam


# 170524_M291236
hemo = hemo_2cam(r"Y:\CorticalMapping\IntrinsicImageData\170531-M287480\M287480_170531_cam1_crosstalkJCamF101.dcimg_4_4_1.h5",
                 r"Y:\CorticalMapping\IntrinsicImageData\170531-M287480\M287480_170531_cam2_crosstalkJCamF108.dcimg_4_4_1.h5",
                 r"Y:\CorticalMapping\IntrinsicImageData\170531-M287480\M287480_170531_cam1_crosstalkJCamF101.dcimg_16_16_1.h5",
                 r"Y:\CorticalMapping\IntrinsicImageData\170531-M287480\M287480_170531_cam2_crosstalkJCamF108.dcimg_16_16_1.h5",
                 r"Y:\CorticalMapping\IntrinsicImageData\170531-M287480\M287480_170531_cam1JCamF103.dcimg_16_16_1.h5",
                 r"Y:\CorticalMapping\IntrinsicImageData\170531-M287480\M287480_170531_cam2JCamF109.dcimg_16_16_1.h5",
                 r"Y:\CorticalMapping\IntrinsicImageData\170531-M287480\M287480_170531_cam1JPhys103",
                 r"G:\170531-M287480\170531_M287480_alldat")

hemo.run_2cam()