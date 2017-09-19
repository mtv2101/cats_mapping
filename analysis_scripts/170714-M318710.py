# -*- coding: utf-8 -*-
"""
Created on Fri Jul 14 14:59:03 2017

@author: mattv
"""

from cats_mapping.hemodynamics.hemo_2cam import hemo_2cam


# 170524_M291236
hemo = hemo_2cam(r"\\allen\programs\braintv\workgroups\nc-ophys\CorticalMapping\IntrinsicImageData\170714-M318710\470nm_cam1_demixtestJCamF118.dcimg_4_4_1.h5",
                 r"\\allen\programs\braintv\workgroups\nc-ophys\CorticalMapping\IntrinsicImageData\170714-M318710\470nm_cam2_demixtestJCamF116.dcimg_4_4_1.h5",
                 r"\\allen\programs\braintv\workgroups\nc-ophys\CorticalMapping\IntrinsicImageData\170714-M318710\470nm_cam1_demixtestJCamF118.dcimg_16_16_1.h5",
                 r"\\allen\programs\braintv\workgroups\nc-ophys\CorticalMapping\IntrinsicImageData\170714-M318710\470nm_cam2_demixtestJCamF116.dcimg_16_16_1.h5",
                 r"\\allen\programs\braintv\workgroups\nc-ophys\CorticalMapping\IntrinsicImageData\170714-M318710\170714-M318710_cam1JCamF101.dcimg_16_16_1.h5",
                 r"\\allen\programs\braintv\workgroups\nc-ophys\CorticalMapping\IntrinsicImageData\170714-M318710\170714-M318710_cam2JCamF118.dcimg_16_16_1.h5",
                 r"\\allen\programs\braintv\workgroups\nc-ophys\CorticalMapping\IntrinsicImageData\170714-M318710\170714-M318710_cam1JPhys101",
                 r"\\allen\programs\braintv\workgroups\nc-ophys\CorticalMapping\IntrinsicImageData\170714-M318710\170714-M318710_alldat")

hemo.run_2cam()