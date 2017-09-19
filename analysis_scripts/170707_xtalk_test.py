# -*- coding: utf-8 -*-
"""
Created on Thu May 25 12:21:57 2017

@author: mattv
"""

from cats_mapping.hemodynamics.hemo_2cam import hemo_2cam


# 170524_M291236
hemo = hemo_2cam(r"Y:\CorticalMapping\IntrinsicImageData\170707_xtalk_test\170707_xtalk_testJcam1.dcimg_4_4_1.h5",
                 r"Y:\CorticalMapping\IntrinsicImageData\170707_xtalk_test\170707_xtalk_test_cam2_4_4_1.h5",
                 r"Y:\CorticalMapping\IntrinsicImageData\170707_xtalk_test\170707_xtalk_test_cam1.dcimg_16_16_1.h5",
                 r"Y:\CorticalMapping\IntrinsicImageData\170707_xtalk_test\170707_xtalk_test_cam2_16_16_1.h5",
                 r"Y:\CorticalMapping\IntrinsicImageData\170707_xtalk_test\170707_xtalk_test_cam1.dcimg_16_16_1.h5",
                 r"Y:\CorticalMapping\IntrinsicImageData\170707_xtalk_test\170707_xtalk_test_cam2_16_16_1.h5",
                 r"Y:\CorticalMapping\IntrinsicImageData\170707_xtalk_test\170707_xtalk_test_cam1_JPhys100",
                 r"Y:\CorticalMapping\IntrinsicImageData\170707_xtalk_test\170707_xtalk_test_cam1_JPhys100",
                 r"Y:\CorticalMapping\IntrinsicImageData\170707_xtalk_test\170707_xtalk_test_alldat")

hemo.run_2cam()