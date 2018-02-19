# -*- coding: utf-8 -*-
"""
Created on Thu May 25 12:21:57 2017

@author: mattv
"""

from cats_mapping.hemodynamics.hemo_2cam import hemo_2cam


# 170524_M291236
hemo = hemo_2cam(r"\\allen\programs\braintv\workgroups\nc-ophys\CorticalMapping\IntrinsicImageData\171009-M287480-DriftingGratingCircle\171009-M287480-cam1-calibJCamF100.dcimg_4_4_1.h5"
        r"\\allen\programs\braintv\workgroups\nc-ophys\CorticalMapping\IntrinsicImageData\171009-M287480-DriftingGratingCircle\171009-M287480-cam2-calibJCamF100.dcimg_4_4_1.h5",
        "\\allen\programs\braintv\workgroups\nc-ophys\CorticalMapping\IntrinsicImageData\171009-M287480-DriftingGratingCircle\171009-M287480-cam1JCamF101.dcimg_16_16_1.h5",
        "\\allen\programs\braintv\workgroups\nc-ophys\CorticalMapping\IntrinsicImageData\171009-M287480-DriftingGratingCircle\171009-M287480-cam2JCamF102.dcimg_16_16_1.h5",
        r"\\allen\programs\braintv\workgroups\nc-ophys\CorticalMapping\IntrinsicImageData\171009-M287480-DriftingGratingCircle\171009-M287480-cam1JCamF101.dcimg_16_16_1.h5",
        r"\\allen\programs\braintv\workgroups\nc-ophys\CorticalMapping\IntrinsicImageData\171009-M287480-DriftingGratingCircle\171009-M287480-cam2JCamF102.dcimg_16_16_1.h5",
        r"\\allen\programs\braintv\workgroups\nc-ophys\CorticalMapping\IntrinsicImageData\171009-M287480-DriftingGratingCircle\171009-M287480-cam1JPhys101",
        r"\\allen\programs\braintv\workgroups\nc-ophys\CorticalMapping\IntrinsicImageData\171009-M287480-DriftingGratingCircle")

hemo.run_2cam()